import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { Cerebras, ConfigError, Groq, Kilo, Mistral, OVHcloud, providersFromEnv } from "../src/index.js";

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

const ENV_KEYS = [
  "OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_KEY",
  "NVIDIA_API_KEY", "NIM_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY", "MISTRAL_API_KEY",
  "FREELM_OPENROUTER_KEYS", "FREELM_GOOGLE_KEYS", "FREELM_NIM_KEYS",
  "FREELM_GROQ_KEYS", "FREELM_CEREBRAS_KEYS", "FREELM_MISTRAL_KEYS",
];
let saved: Record<string, string | undefined> = {};
beforeEach(() => {
  saved = {};
  for (const k of ENV_KEYS) {
    saved[k] = process.env[k];
    delete process.env[k];
  }
});
afterEach(() => {
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
  const vars = ["OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY",
    "CEREBRAS_API_KEY", "MISTRAL_API_KEY", "KILO_API_KEY", "FREELM_KEYLESS"];
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
