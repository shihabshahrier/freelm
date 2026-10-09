import { vi } from "vitest";

export const OK = (content = "hi", model = "m") =>
  JSON.stringify({
    id: "x",
    model,
    choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  });

export interface Call {
  url: string;
  model?: string;
  body: any;
  headers: Record<string, string>;
}

/** Stub global fetch with a handler; returns the recorded calls. */
export function mockFetch(handler: (url: string, body: any, init: any) => Response | Promise<Response>): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: any = {}) => {
      const body = init.body ? JSON.parse(init.body) : undefined;
      calls.push({ url, model: body?.model, body, headers: init.headers ?? {} });
      return handler(url, body, init);
    }) as any,
  );
  return calls;
}

export const sse = (...events: string[]) =>
  new Response(events.join(""), { status: 200, headers: { "content-type": "text/event-stream" } });

export async function collect<T>(it: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const x of it) out.push(x);
  return out;
}
