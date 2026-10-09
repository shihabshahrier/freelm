import { afterEach, beforeEach, expect, it, vi } from "vitest";
import {
  Cerebras,
  CloudflareWorkersAI,
  Cohere,
  ConfigError,
  FreeLLM,
  Groq,
  Kilo,
  Mistral,
  NoProvidersAvailable,
  OVHcloud,
  PROVIDER_ENV,
  providersFromEnv,
  ZAI,
} from "../src/index.js";
import { classify, DAILY_QUOTA_RETRY, ModelNotFound, QuotaExhausted, RateLimited } from "../src/errors.js";
import { collect, mockFetch, OK, sse } from "./helpers.js";

it("new providers construct with correct url/auth/models", () => {
  for (const [P, host] of [
    [Groq, "groq.com"],
    [Cerebras, "cerebras.ai"],
    [Mistral, "mistral.ai"],
  ] as const) {
    const p = new P("key");
    expect(p.url).toContain(host);
    expect(p.url.endsWith("/chat/completions")).toBe(true);
    expect(p.resolveModels("auto").length).toBeGreaterThan(0);
    expect(p.headers("key").Authorization).toBe("Bearer key");
    expect(p.discover).toBe(true);
  }
});

// every variable fromEnv() reads, so a developer's real env can't leak into a test
const ENV_KEYS = [...PROVIDER_ENV.flatMap((s) => [...s.keyVars, ...Object.values(s.optionVars ?? {}).flat()]), "FREELM_KEYLESS"];
let saved: Record<string, string | undefined> = {};
beforeEach(() => {
  saved = {};
  for (const k of ENV_KEYS) {
    saved[k] = process.env[k];
    delete process.env[k];
  }
});
afterEach(() => {
  vi.unstubAllGlobals();
  for (const k of ENV_KEYS) {
    if (saved[k] === undefined) delete process.env[k];
    else process.env[k] = saved[k];
  }
});

it("providersFromEnv picks up the configured providers", () => {
  process.env.GROQ_API_KEY = "gk";
  process.env.CEREBRAS_API_KEY = "ck";
  process.env.MISTRAL_API_KEY = "mk";
  const names = new Set(providersFromEnv().map((p) => p.name));
  expect(names.has("groq")).toBe(true);
  expect(names.has("cerebras")).toBe(true);
  expect(names.has("mistral")).toBe(true);
});

it("keyless providers send no credentials; OVH never takes a (paid) key", () => {
  const k = new Kilo();
  expect(k.headers(k.keys[0].key).Authorization).toBeUndefined();
  expect(k.keys[0].masked()).toBe("(keyless)");
  expect(new Kilo("my-kilo-key").headers("my-kilo-key").Authorization).toBe("Bearer my-kilo-key");
  const o = new OVHcloud();
  expect(o.keys[0].masked()).toBe("(keyless)");
  expect(o.rateLimitScope("")).toBe("model");
});

it("Kilo is free-only", () => {
  const k = new Kilo(null, { discover: false });
  expect(k.resolveModels("poolside/laguna-s-2.1:free")).toEqual(["poolside/laguna-s-2.1:free"]);
  expect(k.resolveModels("kilo-auto/free")).toEqual(["kilo-auto/free"]);
  expect(() => k.resolveModels("anthropic/claude-sonnet-4.5")).toThrow(ConfigError);
});

it("keyless modes: the library never goes keyless on its own", () => {
  const vars = ENV_KEYS;
  const saved = Object.fromEntries(vars.map((v) => [v, process.env[v]]));
  for (const v of vars) delete process.env[v];
  const stderr = vi.spyOn(process.stderr, "write").mockImplementation((() => true) as any);
  try {
    expect(() => providersFromEnv()).toThrow(/FREELM_KEYLESS=1/);
    expect(providersFromEnv("auto").map((p) => p.name)).toEqual(["kilo", "ovh"]);
    process.env.GEMINI_API_KEY = "AIza-x";
    expect(providersFromEnv("auto").map((p) => p.name)).toEqual(["google"]);
    const provs = providersFromEnv(true);
    expect(provs.map((p) => p.name)).toEqual(["google", "kilo", "ovh"]);
    expect(provs[1].priority).toBe(100);
    process.env.FREELM_KEYLESS = "1";
    expect(providersFromEnv().map((p) => p.name)).toEqual(["google", "kilo", "ovh"]);
  } finally {
    stderr.mockRestore();
    for (const v of vars) if (saved[v] === undefined) delete process.env[v]; else process.env[v] = saved[v];
  }
});

// -- Z.ai, Cohere, Cloudflare Workers AI ----------------------------------------------

const CF_ACCOUNT = "0123456789abcdef0123456789abcdef";
const CF_CHAT = `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/v1/chat/completions`;
const ZAI_CHAT = "https://api.z.ai/api/paas/v4/chat/completions";
const cf = (opts = {}) => new CloudflareWorkersAI("cf-token-0123456789", { accountId: CF_ACCOUNT, ...opts });

it("more free providers construct with the right url/auth/models", () => {
  const cases: Array<[any, string]> = [
    [new ZAI("zk"), ZAI_CHAT],
    [new Cohere("ck"), "https://api.cohere.ai/compatibility/v1/chat/completions"],
    [new CloudflareWorkersAI("cft", { accountId: CF_ACCOUNT }), CF_CHAT],
  ];
  for (const [p, url] of cases) {
    expect(p.url).toBe(url);
    expect(p.headers(p.keys[0].key).Authorization).toBe(`Bearer ${p.keys[0].key}`);
    expect(p.resolveModels("auto").length).toBeGreaterThan(0);
    expect(p.discover).toBe(false); // no usable /models list: curated defaults
  }
  // limits are per model on Z.ai and Cohere; Cloudflare's daily allowance is the account's
  expect(new ZAI("k").rateLimitScope("")).toBe("model");
  expect(new Cohere("k").rateLimitScope("")).toBe("model");
  expect(cf().rateLimitScope('{"errors":[{"code":3040,"message":"Capacity temporarily exceeded"}]}')).toBe("model");
  expect(cf().rateLimitScope('{"errors":[{"code":4006,"message":"daily free allocation"}]}')).toBe("key");
});

it("Z.ai only offers its free Flash models", () => {
  const z = new ZAI("k");
  expect(z.resolveModels("vision")).toEqual(["glm-4.6v-flash"]);
  expect(z.resolveModels("glm-4.5-flash")).toEqual(["glm-4.5-flash"]);
  expect(() => z.resolveModels("glm-4.6")).toThrow(/not a free model/); // billed against the balance
});

it("Cloudflare needs a valid account id", () => {
  expect(() => new CloudflareWorkersAI("tok")).toThrow(/CLOUDFLARE_ACCOUNT_ID/);
  expect(() => new CloudflareWorkersAI("tok", { accountId: "my-account" })).toThrow(/32-character/);
  expect(new CloudflareWorkersAI("tok", { accountId: CF_ACCOUNT.toUpperCase() }).accountId).toBe(CF_ACCOUNT);
  // an explicit baseUrl (e.g. an AI Gateway route) needs no account id
  expect(new CloudflareWorkersAI("tok", { baseUrl: "https://gw.example/v1" }).url).toBe("https://gw.example/v1/chat/completions");
});

it("providersFromEnv builds the new providers; a Cloudflare token without account id is skipped", () => {
  const stderr = vi.spyOn(process.stderr, "write").mockImplementation((() => true) as any);
  try {
    process.env.ZAI_API_KEY = "zk";
    process.env.CO_API_KEY = "ck"; // the Cohere SDK's variable works too
    process.env.CLOUDFLARE_API_TOKEN = "cft";
    expect(providersFromEnv().map((p) => p.name)).toEqual(["zai", "cohere"]);
    expect(stderr.mock.calls.map((c) => String(c[0])).join("")).toContain("CLOUDFLARE_ACCOUNT_ID");
    process.env.CLOUDFLARE_ACCOUNT_ID = CF_ACCOUNT;
    const last = providersFromEnv().at(-1)!;
    expect(last.name).toBe("cloudflare");
    expect(last.url).toBe(CF_CHAT);
  } finally {
    stderr.mockRestore();
  }
});

it("Cloudflare requests get room to answer", async () => {
  const model = "@cf/meta/llama-4-scout-17b-16e-instruct";
  const calls = mockFetch(() => new Response(OK("hi", model), { status: 200 }));
  const llm = new FreeLLM([cf()]);
  expect((await llm.chat("hello")).text).toBe("hi");
  await llm.chat("hello", { max_tokens: 50 });
  expect(calls[0].url).toBe(CF_CHAT);
  expect(calls[0].headers.Authorization).toBe("Bearer cf-token-0123456789");
  expect(calls[0].model).toBe(model);
  expect(calls[0].body.max_tokens).toBe(4096); // Workers AI would stop at 256 tokens
  expect(calls[1].body.max_tokens).toBe(50); // the caller's choice wins
});

it("numeric tokens become text", async () => {
  const whole = JSON.parse(OK());
  whole.choices[0].message.content = 6;
  mockFetch((_url, body) =>
    body.stream
      ? sse(
          'data: {"choices":[{"index":0,"delta":{"content":"2 + 4 = "}}]}\n\n',
          'data: {"choices":[{"index":0,"delta":{"content":6}}]}\n\n',
          'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
          "data: [DONE]\n\n",
        )
      : new Response(JSON.stringify(whole), { status: 200 }),
  );
  const llm = new FreeLLM([cf()]);
  expect((await collect(llm.stream("2+4?"))).join("")).toBe("2 + 4 = 6");
  expect((await collect(llm.streamChunks("2+4?")))[1].choices[0].delta.content).toBe("6");
  expect((await llm.chat("2+4?")).text).toBe("6");
});

it("quota 429s rest the key instead of retrying every minute", () => {
  const cfDaily = JSON.stringify({ result: null, success: false, messages: [], errors: [{ code: 4006, message:
    "you have used up your daily free allocation of 10,000 neurons, please upgrade to Cloudflare's Workers Paid plan if you would like to continue usage." }] });
  const e = classify(429, null, cfDaily, "cloudflare");
  expect(e).toBeInstanceOf(RateLimited);
  expect(e.retryAfter).toBe(DAILY_QUOTA_RETRY);
  const orDaily = '{"error":{"message":"Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day","code":429}}';
  expect(classify(429, null, orDaily, "openrouter").retryAfter).toBe(DAILY_QUOTA_RETRY);
  expect(classify(429, { "retry-after": "7" }, orDaily, "openrouter").retryAfter).toBe(7); // a given wait wins
  const perMinute = '{"id":"x","message":"You are using a Trial key, which is limited to 20 API calls / minute."}';
  expect(classify(429, null, perMinute, "cohere").retryAfter).toBeNull(); // the usual short cooldown
  const monthly = '{"id":"x","message":"You are using a Trial key, which is limited to 1000 API calls / month."}';
  expect(classify(429, null, monthly, "cohere")).toBeInstanceOf(QuotaExhausted); // disables the key
});

it("a model missing from the plan benches the model, not the key", () => {
  const paidOnly = '{"errors":[{"code":5035,"message":"This model requires the Workers Paid plan."}],"success":false}';
  const e = classify(403, null, paidOnly, "cloudflare") as ModelNotFound;
  expect(e).toBeInstanceOf(ModelNotFound);
  expect(e.gone).toBe(true);
  expect(e.retired).toBe(false);
  const missing = classify(400, null, '{"errors":[{"code":5007,"message":"No such model @cf/x/y or task"}]}', "cloudflare") as ModelNotFound;
  expect(missing).toBeInstanceOf(ModelNotFound);
  expect(missing.gone).toBe(true);
});

it("Cloudflare's daily allowance rests the key for an hour", async () => {
  const body = { success: false, errors: [{ code: 4006, message: "you have used up your daily free allocation of 10,000 neurons" }] };
  const calls = mockFetch(() => new Response(JSON.stringify(body), { status: 429 }));
  const p = cf();
  await expect(new FreeLLM([p]).chat("hi")).rejects.toBeInstanceOf(NoProvidersAvailable);
  expect(calls.length).toBe(1); // not every model in turn: the account is out
  expect(p.keys[0].cooldownUntil - performance.now() / 1000).toBeGreaterThan(3000);
});

it("Z.ai 429 benches only that model", async () => {
  const calls = mockFetch((_url, body) =>
    body.model === "glm-4.7-flash"
      ? new Response(JSON.stringify({ error: { code: "1302", message: "Rate limit reached for requests" } }), { status: 429 })
      : new Response(OK("ok", body.model), { status: 200 }),
  );
  const p = new ZAI("zk");
  const llm = new FreeLLM([p]);
  expect((await llm.chat("a")).model).toBe("glm-4.5-flash");
  expect((await llm.chat("b")).model).toBe("glm-4.5-flash"); // the benched model isn't retried
  expect(calls.map((c) => c.model)).toEqual(["glm-4.7-flash", "glm-4.5-flash", "glm-4.5-flash"]);
  expect(calls[0].url).toBe(ZAI_CHAT);
  expect(p.keys[0].cooldownUntil).toBe(0); // the key itself stays hot
});
