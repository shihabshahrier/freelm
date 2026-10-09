// Mirrors tests/test_hardening.py, plus TS-only regressions (stream timers,
// AbortSignal, key masking in inspect/JSON).
import { mkdtempSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { inspect } from "node:util";
import { afterEach, describe, expect, it, vi } from "vitest";
import { load as cacheLoad } from "../src/cache.js";
import { sizeTags } from "../src/discovery.js";
import { keyId } from "../src/state.js";
import {
  AuthError,
  BadRequest,
  Cerebras,
  ConfigError,
  FreeLLM,
  GoogleAIStudio,
  Groq,
  ModelNotFound,
  NoProvidersAvailable,
  OpenRouter,
  RateLimited,
  StateStore,
  Transient,
  classify,
  modelSpec,
  resolveModels,
  toSpecs,
} from "../src/index.js";
import { OK, collect, mockFetch, sse } from "./helpers.js";

afterEach(() => vi.unstubAllGlobals());

const isGoogle = (u: string) => u.includes("googleapis");
const isOR = (u: string) => u.includes("openrouter");

describe("error taxonomy", () => {
  it.each([
    [520, "cloudflare unknown error", Transient],
    [524, "a timeout occurred", Transient],
    [501, "not implemented", Transient],
    [400, '{"error":{"message":"API key not valid. Please pass a valid API key.","details":[{"reason":"API_KEY_INVALID"}]}}', AuthError],
    [400, '{"error":{"message":"User location is not supported for the API use."}}', AuthError],
    [403, '{"error":{"message":"Your input was flagged by moderation"}}', BadRequest],
    [413, "Request too large for model `llama` on tokens per minute (TPM): Limit 6000", ModelNotFound],
    [400, "Please reduce the length of the messages or completion", ModelNotFound],
    [451, "unavailable for legal reasons", BadRequest],
    [418, "teapot", BadRequest],
  ])("%i -> %o", (status, body, cls) => {
    expect(classify(status as number, {}, body as string, "p").constructor).toBe(cls);
  });

  it("a capability 404 does not bench the model", () => {
    expect((classify(404, {}, '{"error":{"message":"No endpoints found that support tool use."}}', "openrouter") as ModelNotFound).gone).toBe(false);
    expect((classify(404, {}, '{"error":{"message":"No endpoints found for x/y:free."}}', "openrouter") as ModelNotFound).gone).toBe(true);
  });

  it("Groq 413 (TPM) fails over instead of raising", async () => {
    mockFetch((u) => (u.includes("groq") ? new Response("Request too large ... tokens per minute (TPM)", { status: 413 }) : new Response(OK("g"), { status: 200 })));
    const groq = new Groq("gsk_test", { models: [modelSpec("m", ["chat"])] });
    const llm = new FreeLLM([groq, new GoogleAIStudio("k")]);
    expect((await llm.chat("long prompt")).provider).toBe("google");
    expect(groq.keys[0].disabled).toBe(false);
  });

  it("a moderation 403 does not disable the key", async () => {
    mockFetch((u) => (isOR(u) ? new Response('{"error":{"message":"input was flagged"}}', { status: 403 }) : new Response(OK("g"), { status: 200 })));
    const llm = new FreeLLM([new OpenRouter("k", { discover: false }), new GoogleAIStudio("k2")]);
    expect((await llm.chat("hi")).provider).toBe("google");
    expect(llm.providers[0].keys[0].disabled).toBe(false);
  });

  it("a 402 naming the model benches the model, not the key", async () => {
    const calls = mockFetch((_u, body) =>
      body.model === "gpt-oss-120b"
        ? new Response('{"message":"Payment required to use gpt-oss-120b on the free tier"}', { status: 402 })
        : new Response(OK("free-model"), { status: 200 }),
    );
    const c = new Cerebras("csk-test", { models: [modelSpec("gpt-oss-120b", ["chat"]), modelSpec("qwen", ["chat"])] });
    const llm = new FreeLLM([c]);
    expect((await llm.chat("hi")).text).toBe("free-model");
    expect((await llm.chat("hi")).text).toBe("free-model");
    expect(c.keys[0].disabled).toBe(false);
    expect(calls.map((x) => x.model)).toEqual(["gpt-oss-120b", "qwen", "qwen"]);
  });

  it("an account-wide 402 still disables the key", async () => {
    mockFetch(() => new Response('{"message":"Payment required"}', { status: 402 }));
    const c = new Cerebras("csk-test", { discover: false });
    await expect(new FreeLLM([c]).chat("hi")).rejects.toBeInstanceOf(NoProvidersAvailable);
    expect(c.keys[0].disabled).toBe(true);
  });

  it("Retry-After: 0 means retry now", () => {
    expect((classify(429, { "retry-after": "0" }, "slow", "p") as RateLimited).retryAfter).toBe(0);
  });

  it("bad request rejected by two providers is raised; by one it fails over", async () => {
    const calls = mockFetch(() => new Response("invalid temperature", { status: 400 }));
    const llm = new FreeLLM([new OpenRouter("k1", { discover: false }), new GoogleAIStudio("k2")]);
    const err = await llm.chat("hello").catch((e) => e);
    expect(err).toBeInstanceOf(BadRequest);
    expect(calls.length).toBe(2);
    expect(llm.providers[0].keys[0].breaker.failures).toBe(0);

    mockFetch((u) => (isOR(u) ? new Response("unsupported parameter: seed", { status: 400 }) : new Response(OK("g"), { status: 200 })));
    const llm2 = new FreeLLM([new OpenRouter("k1", { discover: false }), new GoogleAIStudio("k2")]);
    expect((await llm2.chat("hello", { seed: 1 })).provider).toBe("google");
  });

  it("a single provider's rejection is raised as-is after one call", async () => {
    const calls = mockFetch(() => new Response("invalid temperature", { status: 400 }));
    await expect(new FreeLLM([new OpenRouter("k1", { discover: false })]).chat("hello")).rejects.toBeInstanceOf(BadRequest);
    expect(calls.length).toBe(1);
  });

  it("chat() rejects stream: true", async () => {
    await expect(new FreeLLM([new OpenRouter("k", { discover: false })]).chat("hi", { stream: true })).rejects.toBeInstanceOf(ConfigError);
  });
});

describe("secrets", () => {
  it("inspect / JSON / error messages never contain raw keys", async () => {
    const secret = "sk-or-v1-THIS-IS-A-VERY-SECRET-KEY-123456";
    mockFetch(() => new Response("no", { status: 401 }));
    const llm = new FreeLLM([new OpenRouter(secret, { discover: false })]);
    const err = await llm.chat("hi").catch((e) => e);
    const dumps = [inspect(err.attempts, { depth: 4 }), String(err), JSON.stringify(llm.providers[0].keys), inspect(llm.providers[0].keys)];
    for (const d of dumps) expect(d).not.toContain("VERY-SECRET");
  });
});

describe("AbortSignal", () => {
  it("the caller's signal cancels the call without penalising the key", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_u: string, init: any) =>
          new Promise((_, reject) => init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))),
      ) as any,
    );
    const llm = new FreeLLM([new OpenRouter("k", { discover: false })]);
    const ac = new AbortController();
    setTimeout(() => ac.abort(), 30);
    const err = await llm.chat("hi", { signal: ac.signal }).catch((e) => e);
    expect(err.name).toBe("AbortError");
    const k = llm.providers[0].keys[0];
    expect(k.breaker.failures).toBe(0);
    expect(k.cooldownUntil).toBe(0);
  });

  it("signal never reaches the request body", async () => {
    const calls = mockFetch(() => new Response(OK(), { status: 200 }));
    await new FreeLLM([new OpenRouter("k", { discover: false })]).chat("hi", { signal: new AbortController().signal });
    expect("signal" in calls[0].body).toBe(false);
  });
});

describe("stream timers", () => {
  const streamOf = (parts: string[], close: boolean) =>
    new Response(
      new ReadableStream({
        start(c) {
          for (const p of parts) c.enqueue(new TextEncoder().encode(p));
          if (close) c.close();
        },
      }),
      { status: 200 },
    );

  it("a consumer slower than `timeout` neither hangs nor times out", async () => {
    mockFetch(() =>
      streamOf(['data: {"choices":[{"delta":{"content":"a"}}]}\n\n', 'data: {"choices":[{"delta":{"content":"b"}}]}\n\n', "data: [DONE]\n\n"], true),
    );
    const llm = new FreeLLM([new OpenRouter("k", { discover: false })], { timeout: 0.15 });
    const got: string[] = [];
    for await (const c of llm.stream("hi")) {
      got.push(c);
      await new Promise((r) => setTimeout(r, 300)); // longer than the timeout
    }
    expect(got).toEqual(["a", "b"]);
  });

  it("[DONE] ends the stream even if the server keeps the connection open", async () => {
    mockFetch(() => streamOf(['data: {"choices":[{"delta":{"content":"x"}}]}\n\n', "data: [DONE]\n\n"], false));
    const llm = new FreeLLM([new OpenRouter("k", { discover: false })], { timeout: 5 });
    const t0 = performance.now();
    expect((await collect(llm.stream("hi"))).join("")).toBe("x");
    expect(performance.now() - t0).toBeLessThan(1000);
  });

  it("a stalled stream errors out promptly instead of hanging", async () => {
    mockFetch(() => streamOf([], false)); // headers arrive, then nothing ever
    const llm = new FreeLLM([new OpenRouter("k", { discover: false })], { timeout: 0.1 });
    const t0 = performance.now();
    const err = await collect(llm.stream("hi")).catch((e) => e);
    expect(err).toBeInstanceOf(NoProvidersAvailable);
    expect(err.attempts[0][1]).toBeInstanceOf(Transient);
    expect(performance.now() - t0).toBeLessThan(1000);
  });
});

describe("persistence", () => {
  const tmp = () => mkdtempSync(join(tmpdir(), "freelm-state-"));

  it("an expired daily window resets the counter", () => {
    const store = new StateStore(join(tmp(), "state.json"));
    const p = new OpenRouter("sk-or-abc", { discover: false });
    p.keys[0].rpdUsed = 50;
    p.keys[0].rpdReset = 10;
    store.save([p], 20);
    const p2 = new OpenRouter("sk-or-abc", { discover: false });
    store.loadInto([p2], 0);
    expect(p2.keys[0].rpdUsed).toBe(0);
  });

  it("tolerates bad field types and expires old disables", () => {
    const f = join(tmp(), "state.json");
    const id = keyId("openrouter", "k")!;
    writeFileSync(f, JSON.stringify({ [id]: { rpd_used: "n/a", rpd_reset_wall: "x", disabled: true, disabled_since_wall: Date.now() / 1000 - 90000 } }));
    const p = new OpenRouter("k", { discover: false });
    new StateStore(f).loadInto([p], 0);
    expect(p.keys[0].rpdUsed).toBe(0);
    expect(p.keys[0].disabled).toBe(false);
  });

  it("saving leaves no temp files behind", () => {
    const dir = tmp();
    const store = new StateStore(join(dir, "state.json"));
    const p = new OpenRouter("k", { discover: false });
    for (let i = 0; i < 5; i++) store.save([p], 0);
    expect(readdirSync(dir)).toEqual(["state.json"]);
  });

  it("resetKeys gives a clean slate", async () => {
    mockFetch(() => new Response("bad", { status: 401 }));
    const llm = new FreeLLM([new OpenRouter("k", { discover: false })]);
    await expect(llm.chat("hi")).rejects.toBeInstanceOf(NoProvidersAvailable);
    llm.resetKeys();
    expect(llm.health()[0].ready).toBe(true);
  });
});

describe("discovery & model lists", () => {
  it.each([
    ["gemini-2.5-flash", []],
    ["minimax-m2", []],
    ["gpt-4o-mini", ["small", "fast"]],
    ["gemini-2.5-flash-lite", ["small", "fast"]],
    ["nemotron-3-super-120b-a12b", ["large"]],
    ["llama-3.1-405b-instruct", ["large"]],
    ["llama-3.1-8b-instant", ["small", "fast"]],
  ])("sizeTags(%s) = %o", (id, tags) => {
    expect(sizeTags(id as string)).toEqual(tags);
  });

  it("reads Groq/Mistral metadata, strips Google's prefix, drops classifiers", () => {
    const specs = toSpecs(
      [
        { id: "llama-3.3-70b-versatile", context_window: 131072 },
        { id: "mistral-small-latest", max_context_length: 32000, capabilities: { completion_chat: true, function_calling: true, vision: true } },
        { id: "mistral-embed", capabilities: { completion_chat: false } },
        { id: "models/gemini-3.1-flash-lite" },
        { id: "nvidia/nemotron-3.5-content-safety" },
      ],
      false,
    );
    const by = Object.fromEntries(specs.map((s) => [s.id, s]));
    expect(Object.keys(by).sort()).toEqual(["gemini-3.1-flash-lite", "llama-3.3-70b-versatile", "mistral-small-latest"]);
    expect(by["llama-3.3-70b-versatile"].ctx).toBe(131072);
    expect(by["mistral-small-latest"].tags).toEqual(expect.arrayContaining(["tools", "vision"]));
  });

  it("capability aliases don't fall back to incapable models", () => {
    const models = [modelSpec("plain", ["chat"])];
    expect(resolveModels(models, "chat:tools")).toEqual([]);
    expect(resolveModels(models, "vision")).toEqual([]);
    expect(resolveModels(models, "reasoning")).toEqual(["plain"]);
  });

  it("an explicit models list is not overwritten by discovery", () => {
    expect(new OpenRouter("k", { models: [modelSpec("mine:free", ["chat"])] }).discover).toBe(false);
    expect(new Groq("gsk", { models: [modelSpec("m", ["chat"])] }).discover).toBe(false);
    expect(new OpenRouter("k").discover).toBe(true);
  });

  it("discovery skips a dead first key", async () => {
    let models = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: any) => {
        if (url.endsWith("/models")) {
          models++;
          return init.headers.Authorization === "Bearer dead"
            ? new Response("", { status: 401 })
            : new Response(JSON.stringify({ data: [{ id: "fresh/model", context_window: 1000 }] }), { status: 200 });
        }
        return new Response(OK(), { status: 200 });
      }) as any,
    );
    const llm = new FreeLLM([new Groq(["dead", "alive"])]);
    await llm.chat("hi");
    expect(llm.providers[0].models.map((m) => m.id)).toEqual(["fresh/model"]);
    expect(models).toBe(2);
  });

  it("refreshModels bypasses the disk cache; concurrent first calls discover once", async () => {
    let models = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.endsWith("/models")) {
          models++;
          return new Response(JSON.stringify({ data: [{ id: "a/b:free" }] }), { status: 200 });
        }
        return new Response(OK(), { status: 200 });
      }) as any,
    );
    const llm = new FreeLLM([new OpenRouter("k")]);
    await Promise.all(Array.from({ length: 10 }, () => llm.chat("hi")));
    expect(models).toBe(1);
    llm.refreshModels();
    await llm.chat("hi");
    expect(models).toBe(2);
  });

  it("a cache file with the wrong shape is ignored", () => {
    const dir = process.env.FREELM_CACHE_DIR!;
    writeFileSync(join(dir, "models-openrouter.json"), JSON.stringify(["not", "a", "dict"]));
    expect(cacheLoad("openrouter")).toBeNull();
    writeFileSync(join(dir, "models-openrouter.json"), JSON.stringify({ data: "nope", expires_at: 9e12 }));
    expect(cacheLoad("openrouter")).toBeNull();
  });

  it("free-only aliases never resolve to a discovered paid model", async () => {
    const calls = mockFetch((url) =>
      url.endsWith("/models")
        ? new Response(
            JSON.stringify({
              data: [
                { id: "openai/gpt-4o", pricing: { prompt: "0.0000025", completion: "0.00001" } },
                { id: "vendor/free-model:free", pricing: { prompt: "0", completion: "0" } },
              ],
            }),
            { status: 200 },
          )
        : new Response("", { status: 503 }),
    );
    const llm = new FreeLLM([new OpenRouter("k", { discoverFreeOnly: false })]);
    await expect(llm.chat("hi")).rejects.toBeInstanceOf(NoProvidersAvailable);
    expect(new Set(calls.filter((c) => c.model).map((c) => c.model))).toEqual(new Set(["vendor/free-model:free"]));
  });
});

it("Google ids route correctly next to OpenRouter (README example)", async () => {
  mockFetch((u) => new Response(OK(isGoogle(u) ? "g" : "or"), { status: 200 }));
  const llm = new FreeLLM([new OpenRouter("k", { discover: false }), new GoogleAIStudio("k2")]);
  expect((await llm.chat("hi", { model: ["llama-3.3-70b-versatile", "chat:fast"] })).text).toBeTruthy();
  expect(await collect(llm.stream("hi", { model: "gemini-2.5-flash" })).catch((e) => e)).toBeDefined();
});

it("discovery skips audio generators and partially priced entries", () => {
  const specs = toSpecs(
    [
      { id: "google/lyria-3-pro-preview", pricing: { prompt: "0", completion: "0" }, architecture: { output_modalities: ["text", "audio"] } },
      { id: "vendor/img-fee", pricing: { prompt: "0", completion: "0", image: "0.002" } },
      { id: "inclusionai/ling-3.1-flash", pricing: { prompt: "0", completion: "0" }, architecture: { output_modalities: ["text"] } },
      { id: "openrouter/auto", pricing: { prompt: "-1", completion: "-1" } },
    ],
    true,
  );
  expect(specs.map((s) => s.id)).toEqual(["inclusionai/ling-3.1-flash"]);
});
