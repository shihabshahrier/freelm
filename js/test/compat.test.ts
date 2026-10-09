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

// -- OpenAI-SDK fidelity -------------------------------------------------------

const TOOL_PAYLOAD = JSON.stringify({
  id: "chatcmpl-1",
  model: "m",
  choices: [
    {
      index: 0,
      finish_reason: "tool_calls",
      message: {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "call_1", type: "function", function: { name: "get_weather", arguments: '{"city":"Paris"}' } }],
      },
    },
  ],
  usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
});

it("tool-call loop round-trips (message objects passed back, no null fields upstream)", async () => {
  const bodies: any[] = [];
  let n = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_u: string, init: any) => {
      bodies.push(JSON.parse(init.body));
      return new Response(n++ === 0 ? TOOL_PAYLOAD : OK("sunny"), { status: 200 });
    }),
  );
  const client = new OpenAI(new FreeLLM([new OpenRouter("k", { discover: false })]));
  const msgs: any[] = [{ role: "user", content: "weather in Paris?" }];
  const r = await client.chat.completions.create({ model: "auto", messages: msgs, tools: [{ type: "function" }] });
  const call = r.choices[0].message.tool_calls![0];
  expect([call.id, call.function.name, JSON.parse(call.function.arguments)]).toEqual(["call_1", "get_weather", { city: "Paris" }]);
  expect(r.created).toBeGreaterThan(0);
  expect(r.object).toBe("chat.completion");
  msgs.push(r.choices[0].message, { role: "tool", tool_call_id: call.id, content: "sunny" });
  const r2 = await client.chat.completions.create({ model: "auto", messages: msgs });
  expect(r2.choices[0].message.content).toBe("sunny");
  expect("content" in bodies[1].messages[1]).toBe(false); // content: null dropped
});

it("plain completions omit tool_calls instead of sending null", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(OK("x"), { status: 200 })));
  const client = new OpenAI(new FreeLLM([new OpenRouter("k", { discover: false })]));
  const r = await client.chat.completions.create({ model: "auto", messages: [{ role: "user", content: "hi" }] });
  expect("tool_calls" in r.choices[0].message).toBe(false);
});

it("stream chunks carry tool deltas, finish_reason, stable id/created; controller aborts", async () => {
  const sse =
    'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n' +
    'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"f","arguments":"{}"}}]}}]}\n\n' +
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n' +
    "data: [DONE]\n\n";
  vi.stubGlobal("fetch", vi.fn(async () => new Response(sse, { status: 200 })));
  const client = new OpenAI(new FreeLLM([new OpenRouter("k", { discover: false })]));
  const stream = await client.chat.completions.create({ model: "auto", messages: [{ role: "user", content: "hi" }], stream: true });
  expect(stream.controller).toBeInstanceOf(AbortController);
  const chunks = [];
  for await (const c of stream) chunks.push(c);
  expect(new Set(chunks.map((c) => c.id)).size).toBe(1);
  expect(chunks[1].choices[0].delta.tool_calls![0].function.name).toBe("f");
  expect(chunks[chunks.length - 1].choices[0].finish_reason).toBe("tool_calls");
});

it("models.list() resolves to a page and is async-iterable", async () => {
  const client = new OpenAI(new FreeLLM([new OpenRouter("k", { discover: false })]));
  const page = await client.models.list();
  expect(page.data.map((m) => m.id)).toEqual(expect.arrayContaining(["auto", "chat"]));
  const ids: string[] = [];
  for await (const m of client.models.list()) ids.push(m.id);
  expect(ids.some((i) => i.endsWith(":free"))).toBe(true);
});

it("constructor forwards FreeLLM options; openai-node millisecond timeouts are converted", () => {
  process.env.OPENROUTER_API_KEY = "sk-or-env";
  const events: any[] = [];
  const c1 = new OpenAI({ timeout: 30000, onEvent: (e) => events.push(e), strategy: "latency" }) as any;
  expect(c1.client.timeout).toBe(30);
  expect(c1.client.strategy).toBe("latency");
  const c2 = new OpenAI({ timeout: 45 }) as any;
  expect(c2.client.timeout).toBe(45);
});

it("create(body, { signal }) aborts the request", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn((_u: string, init: any) => new Promise((_, rej) => init.signal.addEventListener("abort", () => rej(new DOMException("x", "AbortError"))))),
  );
  const client = new OpenAI(new FreeLLM([new OpenRouter("k", { discover: false })]));
  const ac = new AbortController();
  setTimeout(() => ac.abort(), 20);
  const err = await client.chat.completions.create({ model: "auto", messages: [{ role: "user", content: "hi" }] }, { signal: ac.signal }).catch((e) => e);
  expect(err.name).toBe("AbortError");
});
