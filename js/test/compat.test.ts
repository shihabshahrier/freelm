import { afterEach, expect, it, vi } from "vitest";
import { FreeLLM, OpenRouter } from "../src/index.js";
import { OpenAI } from "../src/compat/openai.js";

const OK = (content = "hi") =>
  JSON.stringify({
    id: "x",
    model: "m",
    choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  });

const SSE = ['data: {"choices":[{"delta":{"content":"Hel"}}]}', "", 'data: {"choices":[{"delta":{"content":"lo"}}]}', "", "data: [DONE]", ""].join("\n");

afterEach(() => {
  vi.unstubAllGlobals();
  delete process.env.OPENROUTER_API_KEY;
});

it("works with an explicit FreeLLM", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(OK("compat"), { status: 200 })));
  const client = new OpenAI(new FreeLLM([new OpenRouter("k", { discover: false })]));
  const r = await client.chat.completions.create({ model: "auto", messages: [{ role: "user", content: "hi" }] });
  expect(r.choices[0].message.content).toBe("compat");
});

it("accepts OpenAI-SDK-style constructor options", async () => {
  process.env.OPENROUTER_API_KEY = "sk-or-env";
  vi.stubGlobal("fetch", vi.fn(async () => new Response(OK("ok"), { status: 200 })));
  // real OpenAI users construct with { apiKey, baseURL, ... } — must not break
  const client = new OpenAI({ apiKey: "sk-ignored", baseURL: "https://api.openai.com/v1", maxRetries: 2 });
  const r = await client.chat.completions.create({ model: "auto", messages: [{ role: "user", content: "hi" }] });
  expect(r.choices[0].message.content).toBe("ok");
});

it("stream: true yields chunk-shaped objects", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(SSE, { status: 200, headers: { "content-type": "text/event-stream" } })),
  );
  const client = new OpenAI(new FreeLLM([new OpenRouter("k", { discover: false })]));
  const stream = await client.chat.completions.create({
    model: "auto",
    messages: [{ role: "user", content: "hi" }],
    stream: true,
  });
  let out = "";
  for await (const chunk of stream) {
    expect(chunk.object).toBe("chat.completion.chunk");
    out += chunk.choices[0].delta.content ?? "";
  }
  expect(out).toBe("Hello");
});
