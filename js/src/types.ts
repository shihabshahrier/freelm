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

  /** Assistant text of the first choice (also via String(resp)). */
  get text(): string {
    return this.choices[0]?.message?.content ?? "";
  }

  toString(): string {
    return this.text;
  }
}

export interface ChatRequest {
  messages: Record<string, any>[];
  model: string; // virtual alias
  params: Record<string, any>; // sampling + passthrough (snake_case, OpenAI-shaped)
}

export function normalizeMessages(input: MessageLike | MessageLike[]): Record<string, any>[] {
  const arr = Array.isArray(input) ? input : [input];
  return arr.map((m) => (typeof m === "string" ? { role: "user", content: m } : (m as Record<string, any>)));
}

export function buildRequest(messages: MessageLike | MessageLike[], model: string, opts: Record<string, any>): ChatRequest {
  return { messages: normalizeMessages(messages), model, params: { ...opts } };
}

export function buildPayload(req: ChatRequest, concreteModel: string): Record<string, any> {
  return { model: concreteModel, messages: req.messages, ...req.params };
}

export function usageFrom(d: any): Usage {
  d = d || {};
  return {
    prompt_tokens: Number(d.prompt_tokens) || 0,
    completion_tokens: Number(d.completion_tokens) || 0,
    total_tokens: Number(d.total_tokens) || 0,
  };
}
