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
 * an async iterable of `chat.completion.chunk` objects (content, tool-call
 * deltas, finish reasons) with a `controller` to abort it.
 *
 * Tools that only speak HTTP (Cursor, Open WebUI, LangChain, the Vercel AI SDK,
 * ...) should use `freelm serve` — the same router as a local OpenAI endpoint.
 */
import { FreeLLM, FreeLLMOptions } from "../client.js";
import { virtualAliases } from "../registry.js";
import { ChatResponse, chunkStamper, completionBody } from "../types.js";

export interface CompatCompletion {
  id: string;
  object: "chat.completion";
  created: number;
  model: string | null;
  provider: string | null;
  choices: Array<{ index: number; message: { role: string; content: string | null; tool_calls?: any[] }; finish_reason: string | null; [k: string]: any }>;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number; [k: string]: any };
  [k: string]: any;
}

export interface CompatChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string | null;
  choices: Array<{ index: number; delta: { role?: string; content?: string | null; tool_calls?: any[]; [k: string]: any }; finish_reason: string | null; [k: string]: any }>;
  [k: string]: any;
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
  dangerouslyAllowBrowser?: boolean;
  [key: string]: any;
}

/** Per-request options (the SDK's second `create()` argument). */
export interface RequestOptions {
  signal?: AbortSignal;
  [key: string]: any;
}

function toFreeLLMOptions(opts: OpenAIOptions): FreeLLMOptions {
  const { strategy, maxAttempts, timeout, wait, maxWait, onEvent, persist } = opts;
  const out: FreeLLMOptions = {};
  if (strategy !== undefined) out.strategy = strategy;
  if (maxAttempts !== undefined) out.maxAttempts = maxAttempts;
  // openai-node takes milliseconds (default 600000), FreeLLM seconds: values
  // above 1000 can only be milliseconds
  if (typeof timeout === "number") out.timeout = timeout > 1000 ? timeout / 1000 : timeout;
  if (wait !== undefined) out.wait = wait;
  if (maxWait !== undefined) out.maxWait = maxWait;
  if (onEvent !== undefined) out.onEvent = onEvent;
  if (persist !== undefined) out.persist = persist;
  return out;
}

/** Duck-typed check: a CommonJS consumer may hold a FreeLLM from a different
 * bundle copy, where `instanceof` would wrongly fail. */
function isFreeLLM(x: any): x is FreeLLM {
  return !!x && typeof x.chat === "function" && typeof x.streamChunks === "function" && Array.isArray(x.providers);
}

function wrap(resp: ChatResponse): CompatCompletion {
  return completionBody(resp) as CompatCompletion;
}

/** Linked abort: fires when either the caller's signal or the stream's own
 * controller aborts. */
function anySignal(a?: AbortSignal, b?: AbortSignal): AbortSignal | undefined {
  if (!a) return b;
  if (!b) return a;
  const ac = new AbortController();
  const fire = () => ac.abort();
  if (a.aborted || b.aborted) ac.abort();
  a.addEventListener("abort", fire, { once: true });
  b.addEventListener("abort", fire, { once: true });
  return ac.signal;
}

/** Async iterable of chat.completion.chunk objects, like openai-node's Stream. */
export class CompatStream implements AsyncIterable<CompatChunk> {
  controller = new AbortController();
  private stamp = chunkStamper();

  constructor(private open: (signal: AbortSignal) => AsyncGenerator<Record<string, any>>) {}

  async *[Symbol.asyncIterator](): AsyncGenerator<CompatChunk> {
    for await (const c of this.open(this.controller.signal)) yield this.stamp(c) as CompatChunk;
  }
}

type CreateArgs = { model?: string | string[]; messages?: any[]; stream?: boolean | null; stream_options?: any; [k: string]: any };

class Completions {
  constructor(private client: FreeLLM) {}

  create(args: CreateArgs & { stream: true }, options?: RequestOptions): Promise<CompatStream>;
  create(args?: CreateArgs & { stream?: false | null }, options?: RequestOptions): Promise<CompatCompletion>;
  async create(args: CreateArgs = {}, options: RequestOptions = {}): Promise<CompatCompletion | CompatStream> {
    const { model = "auto", messages = [], stream, stream_options: _so, ...rest } = args;
    const params: Record<string, any> = {};
    for (const [k, v] of Object.entries(rest)) if (v !== null && v !== undefined) params[k] = v;
    if (stream) {
      return new CompatStream((own) =>
        this.client.streamChunks(messages, { ...params, model, signal: anySignal(options.signal, own) }),
      );
    }
    return wrap(await this.client.chat(messages, { ...params, model, signal: options.signal }));
  }
}

class Chat {
  completions: Completions;
  constructor(client: FreeLLM) {
    this.completions = new Completions(client);
  }
}

export interface CompatModel {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
}

/** `await client.models.list()` -> { object: "list", data }, and `for await
 * (const m of client.models.list())` iterates the models, like openai-node. */
class Models {
  constructor(private client: FreeLLM) {}

  list(): Promise<{ object: "list"; data: CompatModel[] }> & AsyncIterable<CompatModel> {
    const page = (async () => {
      await (this.client as any).ensureDiscovered?.();
      const created = Math.floor(Date.now() / 1000);
      const data: CompatModel[] = virtualAliases().map((id) => ({ id, object: "model", created, owned_by: "freelm" }));
      const seen = new Set(data.map((d) => d.id));
      for (const p of this.client.providers)
        for (const m of p.models)
          if (!seen.has(m.id)) {
            seen.add(m.id);
            data.push({ id: m.id, object: "model", created, owned_by: p.name });
          }
      return { object: "list" as const, data };
    })();
    return Object.assign(page, {
      async *[Symbol.asyncIterator]() {
        yield* (await page).data;
      },
    });
  }
}

export class OpenAI {
  chat: Chat;
  models: Models;
  private client: FreeLLM;
  /** Accepts a FreeLLM instance, OpenAI-SDK-style options, or nothing. */
  constructor(clientOrOpts?: FreeLLM | OpenAIOptions) {
    this.client = isFreeLLM(clientOrOpts) ? clientOrOpts : FreeLLM.fromEnv(toFreeLLMOptions((clientOrOpts as OpenAIOptions) ?? {}));
    this.chat = new Chat(this.client);
    this.models = new Models(this.client);
  }

  withOptions(_opts: Record<string, any> = {}): OpenAI {
    return this; // per-request transport options don't apply to freelm
  }
}

export default OpenAI;
