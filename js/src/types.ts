/** Provider-agnostic types (OpenAI-shaped). */

export type MessageLike = string | Message | Record<string, any>;

export interface Message {
  role: string;
  content: string | null;
  name?: string;
  tool_calls?: any[] | null;
  tool_call_id?: string;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface Choice {
  index: number;
  message: Message;
  finish_reason: string | null;
}

export class ChatResponse {
  constructor(
    public id: string | null,
    public model: string | null,
    public provider: string | null,
    public choices: Choice[],
    public usage: Usage,
    public latencyMs = 0,
    public raw: any = null,
  ) {}

  /** Assistant text of the first choice (also via String(resp)). Content-part
   * arrays ([{type: "text", text}, ...]) are flattened to their text. */
  get text(): string {
    const content: any = this.choices[0]?.message?.content;
    if (Array.isArray(content)) {
      return content.filter((p) => p && p.type === "text").map((p) => p.text ?? "").join("");
    }
    return content ?? "";
  }

  /** Tool calls of the first choice, if the model requested any. */
  get toolCalls(): any[] | null {
    return this.choices[0]?.message?.tool_calls ?? null;
  }

  toString(): string {
    return this.text;
  }
}

/** One observability event emitted via `new FreeLLM(provs, { onEvent })`.
 * `key` is always masked — never a raw API key. */
export interface FreeLLMEvent {
  kind: "attempt" | "success" | "error" | "wait" | "discovery";
  provider: string | null;
  key: string | null;
  model: string | null;
  status: number | null;
  latencyMs: number | null;
  error: string | null;
  attempt: number;
}

export interface ChatRequest {
  messages: Record<string, any>[];
  model: string | string[]; // alias, or ordered fallback chain
  params: Record<string, any>; // sampling + passthrough (snake_case, OpenAI-shaped)
}

/** Normalize messages for the wire: a string is a user turn; objects keep every
 * field except null/undefined ones — providers reject e.g. `"tool_calls": null`
 * (which the OpenAI SDK's own response messages carry). */
export function normalizeMessages(input: MessageLike | MessageLike[]): Record<string, any>[] {
  const arr = Array.isArray(input) ? input : [input];
  return arr.map((m) => {
    if (typeof m === "string") return { role: "user", content: m };
    const out: Record<string, any> = {};
    for (const [k, v] of Object.entries(m as Record<string, any>)) if (v !== null && v !== undefined) out[k] = v;
    out.role ??= "user";
    return out;
  });
}

export function buildRequest(
  messages: MessageLike | MessageLike[],
  model: string | string[],
  opts: Record<string, any>,
): ChatRequest {
  return { messages: normalizeMessages(messages), model, params: { ...opts } };
}

export function buildPayload(req: ChatRequest, concreteModel: string): Record<string, any> {
  return { model: concreteModel, messages: req.messages, ...req.params };
}

const completionId = () => `chatcmpl-${Math.random().toString(36).slice(2, 14)}${Date.now().toString(36)}`;
const nowSec = () => Math.floor(Date.now() / 1000);

/** A `chat.completion` object from a freelm response (provider JSON kept,
 * missing id/created/message fields filled, `tool_calls: null` dropped). */
export function completionBody(resp: ChatResponse): Record<string, any> {
  const raw: any = resp.raw && typeof resp.raw === "object" ? { ...resp.raw } : {};
  const choices = (Array.isArray(raw.choices) && raw.choices.length ? raw.choices : resp.choices).map((c: any) => {
    const m = { ...(c.message ?? {}) };
    m.role ??= "assistant";
    m.content ??= null;
    if (m.tool_calls == null) delete m.tool_calls; // never echo `tool_calls: null` back upstream
    return { ...c, index: c.index ?? 0, message: m, finish_reason: c.finish_reason ?? null };
  });
  return {
    ...raw,
    id: raw.id || resp.id || completionId(),
    object: "chat.completion",
    created: raw.created || nowSec(),
    model: raw.model ?? resp.model,
    provider: resp.provider, // which free tier served it (non-standard, handy)
    choices,
    usage: raw.usage ?? resp.usage,
  };
}

/** Stamps every chunk of one stream with the same id/created, like the SDK. */
export function chunkStamper(): (c: Record<string, any>) => Record<string, any> {
  const id = completionId();
  const created = nowSec();
  return (c) => {
    const choices = (Array.isArray(c.choices) ? c.choices : []).map((ch: any) => ({
      ...ch,
      index: ch?.index ?? 0,
      delta: ch?.delta ?? {},
      finish_reason: ch?.finish_reason ?? null,
    }));
    return { ...c, id: c.id || id, object: "chat.completion.chunk", created: c.created || created, model: c.model ?? null, choices };
  };
}

export function usageFrom(d: any): Usage {
  d = d || {};
  return {
    prompt_tokens: Number(d.prompt_tokens) || 0,
    completion_tokens: Number(d.completion_tokens) || 0,
    total_tokens: Number(d.total_tokens) || 0,
  };
}
