// Mirrors tests/test_reliability.py: failure modes seen against the live free
// tiers (2026-10) — retired models, per-model quotas/overloads, routing of
// concrete ids past the free guard, errors delivered with HTTP 200, raw chunk
// streaming and self-explaining exhaustion errors.
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  AuthError,
  ConfigError,
  FreeLLM,
  GoogleAIStudio,
  ModelNotFound,
  NIM,
  NoProvidersAvailable,
  OpenRouter,
  RateLimited,
  Transient,
  classify,
  modelSpec,
} from "../src/index.js";
import { OK, collect, mockFetch, sse } from "./helpers.js";

afterEach(() => vi.unstubAllGlobals());

const GONE = JSON.stringify({ status: 410, title: "Gone", detail: "The model has reached its end of life" });
const isNim = (u: string) => u.includes("nvidia");
const isGoogle = (u: string) => u.includes("googleapis");
const isOR = (u: string) => u.includes("openrouter");
const nim = (models: string[]) => new NIM("nvapi-test", { models: models.map((m) => modelSpec(m, ["chat"])) });

describe("classification", () => {
  it.each([
    [404, "not found", true],
    [410, GONE, true],
    [400, '{"error":{"message":"The model `x` has been decommissioned","code":"model_decommissioned"}}', true],
    [400, "This model's maximum context length is 8192 tokens", false],
    [413, "Request too large for model llama on tokens per minute", false],
    [422, "model: field required", false],
  ])("%i %s -> ModelNotFound(gone=%s)", (status, body, gone) => {
    const err = classify(status as number, {}, body as string, "p");
    expect(err).toBeInstanceOf(ModelNotFound);
    expect((err as ModelNotFound).gone).toBe(gone);
  });

  it("parses Google's retryDelay; a Retry-After header wins", () => {
    const body = '{"error":{"code":429,"details":[{"retryDelay":"36s"}]}}';
    expect((classify(429, {}, body, "google") as RateLimited).retryAfter).toBe(36);
    expect(classify(429, { "retry-after": "5" }, body, "google").retryAfter).toBe(5);
  });
});

describe("retired models (the 2026-08 NIM end-of-life incident)", () => {
  it("410 Gone fails over instead of raising, and benches the model not the key", async () => {
    mockFetch((u) => (isNim(u) ? new Response(GONE, { status: 410 }) : new Response(OK("from-google"), { status: 200 })));
    const llm = new FreeLLM([new NIM("nvapi-test"), new GoogleAIStudio("k")], { strategy: "priority" });
    const r = await llm.chat("hello");
    expect(r.provider).toBe("google");
    const p = llm.providers[0];
    expect(p.keys[0].disabled).toBe(false);
    expect(p._modelUntil.has(p.models[0].id)).toBe(true);
  });

  it("a retired model is benched across calls", async () => {
    const calls = mockFetch((_u, body) =>
      body.model === "dead/model" ? new Response(GONE, { status: 410 }) : new Response(OK("alive"), { status: 200 }),
    );
    const llm = new FreeLLM([nim(["dead/model", "alive/model"])]);
    expect((await llm.chat("one")).text).toBe("alive");
    expect((await llm.chat("two")).text).toBe("alive");
    expect(calls.map((c) => c.model)).toEqual(["dead/model", "alive/model", "alive/model"]);
  });

  it("dead models do not starve the interleave", async () => {
    const calls = mockFetch((u) => (isNim(u) ? new Response(GONE, { status: 410 }) : new Response(OK("g"), { status: 200 })));
    const llm = new FreeLLM([nim(["a", "b", "c", "d"]), new GoogleAIStudio("k")], { strategy: "priority" });
    expect((await llm.chat("hello")).provider).toBe("google");
    expect(calls.filter((c) => isNim(c.url)).length).toBe(1);
  });

  it("410 during stream fails over", async () => {
    mockFetch((u) =>
      isNim(u) ? new Response(GONE, { status: 410 }) : sse('data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', "data: [DONE]\n\n"),
    );
    const llm = new FreeLLM([new NIM("nvapi-test"), new GoogleAIStudio("k")]);
    expect((await collect(llm.stream("hi"))).join("")).toBe("ok");
  });
});

describe("concrete ids + the OpenRouter free guard", () => {
  it("a known Google id is not blocked by OpenRouter's guard", async () => {
    const calls = mockFetch((u) => new Response(OK(isGoogle(u) ? "g" : "or"), { status: 200 }));
    const llm = new FreeLLM([new OpenRouter("k", { discover: false }), new GoogleAIStudio("k2")]);
    const r = await llm.chat("hi", { model: "gemini-2.5-flash" });
    expect(r.provider).toBe("google");
    expect(calls.filter((c) => isOR(c.url)).length).toBe(0);
    expect(calls[0].model).toBe("gemini-2.5-flash");
  });

  it("a concrete id routes only to providers that list it", async () => {
    const calls = mockFetch(() => new Response(OK(), { status: 200 }));
    const or = new OpenRouter("k", { discover: false, models: [modelSpec("vendor/m:free", ["chat"])] });
    const llm = new FreeLLM([new GoogleAIStudio("k2"), or]);
    expect((await llm.chat("hi", { model: "vendor/m:free" })).provider).toBe("openrouter");
    expect(calls.length).toBe(1);
  });

  it("an unknown paid id skips the guarded provider but reaches others", async () => {
    const calls = mockFetch(() => new Response(OK("g"), { status: 200 }));
    const llm = new FreeLLM([new OpenRouter("k", { discover: false }), new GoogleAIStudio("k2")]);
    expect((await llm.chat("hi", { model: "some-vendor/unlisted-model" })).provider).toBe("google");
    expect(calls.length).toBe(1);
  });

  it("the guard still throws when nothing else can serve", async () => {
    const llm = new FreeLLM([new OpenRouter("k", { discover: false })]);
    await expect(llm.chat("hi", { model: "openai/gpt-5" })).rejects.toBeInstanceOf(ConfigError);
  });

  it("the guard skips one alias of a fallback chain", async () => {
    const calls = mockFetch(() => new Response(OK("ok"), { status: 200 }));
    const llm = new FreeLLM([new OpenRouter("k", { discover: false })]);
    expect((await llm.chat("hi", { model: ["openai/gpt-5", "chat"] })).text).toBe("ok");
    expect(calls[0].model?.endsWith(":free")).toBe(true);
  });
});

describe("Google per-model quotas and overloads", () => {
  const two = () => new GoogleAIStudio("k", { models: [modelSpec("m1", ["chat"]), modelSpec("m2", ["chat"])] });

  it("429 is model-scoped", async () => {
    const calls = mockFetch((_u, body) =>
      body.model === "m1" ? new Response('{"error":{"code":429,"message":"quota, model: m1"}}', { status: 429 }) : new Response(OK("m2-ok"), { status: 200 }),
    );
    const llm = new FreeLLM([two()]);
    expect((await llm.chat("one")).text).toBe("m2-ok");
    expect(llm.providers[0].keys[0].cooldownUntil).toBe(0);
    expect((await llm.chat("two")).text).toBe("m2-ok");
    expect(calls.map((c) => c.model)).toEqual(["m1", "m2", "m2"]);
  });

  it("503 'high demand' benches only the model", async () => {
    mockFetch((_u, body) =>
      body.model === "m1"
        ? new Response('{"error":{"code":503,"message":"This model is currently experiencing high demand."}}', { status: 503 })
        : new Response(OK("m2-ok"), { status: 200 }),
    );
    const llm = new FreeLLM([two()]);
    expect((await llm.chat("hi")).text).toBe("m2-ok");
    expect(llm.providers[0].keys[0].cooldownUntil).toBe(0);
    expect(llm.providers[0].keys[0].breaker.failures).toBe(0);
  });

  it("a generic 503 still cools the key", async () => {
    mockFetch(() => new Response("upstream connect error", { status: 503 }));
    const llm = new FreeLLM([two()]);
    await expect(llm.chat("hi")).rejects.toBeInstanceOf(NoProvidersAvailable);
    expect(llm.providers[0].keys[0].cooldownUntil).toBeGreaterThan(0);
  });
});

describe("errors delivered with HTTP 200", () => {
  const pair = () => [new OpenRouter("k", { discover: false }), new GoogleAIStudio("k2")];

  it("invalid JSON fails over", async () => {
    mockFetch((u) => (isOR(u) ? new Response("<html>bad gateway</html>", { status: 200 }) : new Response(OK("g"), { status: 200 })));
    expect((await new FreeLLM(pair()).chat("hi")).provider).toBe("google");
  });

  it("an error object fails over", async () => {
    mockFetch((u) =>
      isOR(u) ? new Response('{"error":{"code":429,"message":"slow down"}}', { status: 200 }) : new Response(OK("g"), { status: 200 }),
    );
    expect((await new FreeLLM(pair()).chat("hi")).provider).toBe("google");
  });

  it("an error frame before the first token fails over", async () => {
    mockFetch((u) =>
      isOR(u)
        ? sse('data: {"error":{"code":502,"message":"upstream died"}}\n\n')
        : sse('data: {"choices":[{"delta":{"content":"fine"}}]}\n\n', "data: [DONE]\n\n"),
    );
    expect((await collect(new FreeLLM(pair()).stream("hi"))).join("")).toBe("fine");
  });

  it("OpenRouter's mid-stream error shape (error + choices) fails over", async () => {
    mockFetch((u) =>
      isOR(u)
        ? sse(
            'data: {"error":{"code":"server_error","message":"Provider disconnected"},"choices":[{"index":0,"delta":{"content":""},"finish_reason":"error"}]}\n\n',
          )
        : sse('data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', "data: [DONE]\n\n"),
    );
    expect((await collect(new FreeLLM(pair()).stream("hi"))).join("")).toBe("ok");
  });

  it("an error frame after the first token raises (no silent splice)", async () => {
    mockFetch((u) =>
      isOR(u)
        ? sse('data: {"choices":[{"delta":{"content":"par"}}]}\n\n', 'data: {"error":{"code":500,"message":"died"}}\n\n')
        : new Response(OK("g"), { status: 200 }),
    );
    const got: string[] = [];
    await expect(
      (async () => {
        for await (const c of new FreeLLM(pair()).stream("hi")) got.push(c);
      })(),
    ).rejects.toBeInstanceOf(Transient);
    expect(got).toEqual(["par"]);
  });

  it("SSE decoder handles comments, multi-line data, CR endings and [DONE]", async () => {
    mockFetch(() =>
      sse(
        ": keep-alive\r\r",
        'data: {"choices":[{"delta":\r',
        'data: {"content":"A"}}]}\r\r',
        'event: message\ndata:{"choices":[{"delta":{"content":"B"}}]}\n\n',
        "data: [DONE]\n\n",
        'data: {"choices":[{"delta":{"content":"after-done"}}]}\n\n',
      ),
    );
    const llm = new FreeLLM([new OpenRouter("k", { discover: false })]);
    expect((await collect(llm.stream("hi"))).join("")).toBe("AB");
  });
});

const TOOL_SSE = [
  'data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
  'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"get_weather","arguments":"{\\"city\\":"}}]}}]}\n\n',
  'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"Paris\\"}"}}]}}]}\n\n',
  'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
  "data: [DONE]\n\n",
];

describe("raw chunk streaming", () => {
  it("streamChunks keeps tool calls and the finish reason", async () => {
    mockFetch(() => sse(...TOOL_SSE));
    const llm = new FreeLLM([new OpenRouter("k", { discover: false })]);
    const chunks = await collect(llm.streamChunks("weather?", { tools: [{ type: "function" }] }));
    expect(chunks.length).toBe(4);
    expect(chunks[0].choices[0].delta).toEqual({ role: "assistant" });
    const args = chunks.slice(1, 3).map((c) => c.choices[0].delta.tool_calls[0].function.arguments).join("");
    expect(JSON.parse(args)).toEqual({ city: "Paris" });
    expect(chunks[3].choices[0].finish_reason).toBe("tool_calls");
  });

  it("the preamble is held back during failover", async () => {
    mockFetch((u) =>
      isOR(u)
        ? sse('data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n', 'data: {"error":{"code":503,"message":"x"}}\n\n')
        : sse(...TOOL_SSE),
    );
    const llm = new FreeLLM([new OpenRouter("k", { discover: false }), new GoogleAIStudio("k2")]);
    expect((await collect(llm.streamChunks("hi"))).length).toBe(4);
  });
});

describe("exhaustion", () => {
  it("wait mode gives up promptly when waiting cannot help", async () => {
    mockFetch(() => new Response(GONE, { status: 410 }));
    const llm = new FreeLLM([nim(["a"])], { wait: true, maxWait: 20, timeout: 30 });
    const t0 = performance.now();
    await expect(llm.chat("hi")).rejects.toBeInstanceOf(NoProvidersAvailable);
    expect(performance.now() - t0).toBeLessThan(2000);
  });

  it("the error explains each provider and points at `freelm doctor`", async () => {
    mockFetch((u) =>
      isOR(u)
        ? new Response('{"error":{"message":"User not found."}}', { status: 401 })
        : new Response("Resource exhausted", { status: 429, headers: { "retry-after": "30" } }),
    );
    const llm = new FreeLLM([
      new OpenRouter("k", { discover: false }),
      new GoogleAIStudio("k2", { models: [modelSpec("m1", ["chat"])] }),
    ]);
    const err = await llm.chat("hi").catch((e) => e);
    expect(err).toBeInstanceOf(NoProvidersAvailable);
    expect(err.message).toContain("openrouter: disabled (auth:401 — key invalid/expired?)");
    expect(err.message).toContain("freelm doctor");
    expect(err.attempts.some(([, e]: [any, Error]) => e instanceof AuthError)).toBe(true);
    expect(err.status.length).toBeGreaterThan(0);
  });
});
