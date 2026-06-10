/** Drop-in OpenAI-style shim backed by FreeLLM.
 *
 *   // import OpenAI from "openai";
 *   import { OpenAI } from "freelm/compat";
 *   const client = new OpenAI();            // FreeLLM.fromEnv()
 *   const r = await client.chat.completions.create({
 *     model: "auto",
 *     messages: [{ role: "user", content: "hi" }],
 *   });
 *   console.log(r.choices[0].message.content);
 *
 * OpenAI-SDK constructor options ({ apiKey, baseURL, ... }) are accepted and
 * ignored — keys come from the environment / providers. `stream: true` returns
 * an async iterable of `chat.completion.chunk`-shaped objects.
 */
import { FreeLLM, FreeLLMOptions } from "../client.js";
import { ChatResponse } from "../types.js";

export interface CompatCompletion {
  id: string | null;
  object: "chat.completion";
  model: string | null;
  provider: string | null;
  choices: Array<{ index: number; message: { role: string; content: string | null; tool_calls?: any[] | null }; finish_reason: string | null }>;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

export interface CompatChunk {
  object: "chat.completion.chunk";
  choices: Array<{ index: number; delta: { content?: string }; finish_reason: string | null }>;
}

/** OpenAI-SDK client options, accepted for drop-in compatibility (unused),
 * plus the FreeLLM options that actually take effect. */
export interface OpenAIOptions extends FreeLLMOptions {
  apiKey?: string;
  baseURL?: string;
  organization?: string;
  project?: string;
  maxRetries?: number;
  defaultHeaders?: Record<string, string>;
  defaultQuery?: Record<string, string>;
  fetch?: unknown;
  [key: string]: any;
}

function toFreeLLMOptions(opts: OpenAIOptions): FreeLLMOptions {
  const { strategy, maxAttempts, timeout, wait, maxWait } = opts;
  const out: FreeLLMOptions = {};
  if (strategy !== undefined) out.strategy = strategy;
  if (maxAttempts !== undefined) out.maxAttempts = maxAttempts;
  if (typeof timeout === "number") out.timeout = timeout;
  if (wait !== undefined) out.wait = wait;
  if (maxWait !== undefined) out.maxWait = maxWait;
  return out;
}

function wrap(resp: ChatResponse): CompatCompletion {
  return {
    id: resp.id,
    object: "chat.completion",
    model: resp.model,
    provider: resp.provider,
    choices: resp.choices.map((c) => ({
      index: c.index,
      message: { role: c.message.role, content: c.message.content, tool_calls: c.message.tool_calls },
      finish_reason: c.finish_reason,
    })),
    usage: resp.usage,
  };
}

async function* wrapStream(deltas: AsyncGenerator<string>): AsyncGenerator<CompatChunk> {
  for await (const content of deltas) {
    yield { object: "chat.completion.chunk", choices: [{ index: 0, delta: { content }, finish_reason: null }] };
  }
}

type CreateArgs = { model?: string; messages?: any[]; stream?: boolean; stream_options?: any; [k: string]: any };

class Completions {
  constructor(private client: FreeLLM) {}

  create(args: CreateArgs & { stream: true }): Promise<AsyncGenerator<CompatChunk>>;
  create(args?: CreateArgs & { stream?: false }): Promise<CompatCompletion>;
  async create(args: CreateArgs = {}): Promise<CompatCompletion | AsyncGenerator<CompatChunk>> {
    const { model = "auto", messages = [], stream, stream_options: _so, ...rest } = args;
    if (stream) return wrapStream(this.client.stream(messages, { model, ...rest }));
    const resp = await this.client.chat(messages, { model, ...rest });
    return wrap(resp);
  }
}

class Chat {
  completions: Completions;
  constructor(client: FreeLLM) {
    this.completions = new Completions(client);
  }
}

export class OpenAI {
  chat: Chat;
  private client: FreeLLM;
  /** Accepts a FreeLLM instance, OpenAI-SDK-style options, or nothing. */
  constructor(clientOrOpts?: FreeLLM | OpenAIOptions) {
    this.client = clientOrOpts instanceof FreeLLM ? clientOrOpts : FreeLLM.fromEnv(toFreeLLMOptions(clientOrOpts ?? {}));
    this.chat = new Chat(this.client);
  }
}
