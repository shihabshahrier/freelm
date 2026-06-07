/** Drop-in OpenAI-style shim backed by FreeLLM.
 *
 *   import { OpenAI } from "freelm/compat";
 *   const client = new OpenAI();            // FreeLLM.fromEnv()
 *   const r = await client.chat.completions.create({
 *     model: "auto",
 *     messages: [{ role: "user", content: "hi" }],
 *   });
 *   console.log(r.choices[0].message.content);
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

class Completions {
  constructor(private client: FreeLLM) {}
  async create(args: { model?: string; messages?: any[]; stream?: boolean; [k: string]: any }): Promise<CompatCompletion> {
    const { model = "auto", messages = [], stream: _stream, ...rest } = args;
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
  constructor(client?: FreeLLM, opts: FreeLLMOptions = {}) {
    this.client = client ?? FreeLLM.fromEnv(opts);
    this.chat = new Chat(this.client);
  }
}
