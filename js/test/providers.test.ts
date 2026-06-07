import { afterEach, beforeEach, expect, it } from "vitest";
import { Cerebras, Groq, Mistral, providersFromEnv } from "../src/index.js";

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
