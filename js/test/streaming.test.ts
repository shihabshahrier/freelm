import { afterEach, expect, it, vi } from "vitest";
import { FreeLLM, GoogleAIStudio, OpenRouter } from "../src/index.js";

const SSE =
  'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n' +
  'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n' +
  'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n' +
  "data: [DONE]\n\n";

function mockFetch(handler: (url: string, init: any) => Promise<Response>) {
  vi.stubGlobal("fetch", vi.fn(handler as any));
}
afterEach(() => vi.unstubAllGlobals());

it("streams content deltas", async () => {
  mockFetch(async () => new Response(SSE, { status: 200, headers: { "content-type": "text/event-stream" } }));
  const llm = new FreeLLM([new OpenRouter("k", { discover: false })]);
  let out = "";
  for await (const chunk of llm.stream("hi")) out += chunk;
  expect(out).toBe("Hello");
});

it("fails over before the first token", async () => {
  mockFetch(async (url) =>
    url.includes("openrouter") ? new Response("account rate limit", { status: 429 }) : new Response(SSE, { status: 200 }),
  );
  const llm = new FreeLLM([new OpenRouter("k", { discover: false }), new GoogleAIStudio("k2")]);
  let out = "";
  for await (const chunk of llm.stream("hi")) out += chunk;
  expect(out).toBe("Hello");
});
