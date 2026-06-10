import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { main } from "../src/cli.js";

const OK = JSON.stringify({
  id: "x",
  model: "m",
  choices: [{ index: 0, message: { role: "assistant", content: "pong" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
});

const KEY_VARS = [
  "OPENROUTER_API_KEY", "FREELM_OPENROUTER_KEYS",
  "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_KEY", "FREELM_GOOGLE_KEYS",
  "NVIDIA_API_KEY", "NIM_API_KEY", "FREELM_NIM_KEYS",
  "GROQ_API_KEY", "FREELM_GROQ_KEYS",
  "CEREBRAS_API_KEY", "FREELM_CEREBRAS_KEYS",
  "MISTRAL_API_KEY", "FREELM_MISTRAL_KEYS",
];

let out: string[];
let err: string[];

beforeEach(() => {
  process.env.FREELM_CACHE_DIR = mkdtempSync(join(tmpdir(), "freelm-cli-"));
  for (const v of KEY_VARS) delete process.env[v];
  out = [];
  err = [];
  vi.spyOn(process.stdout, "write").mockImplementation(((s: any) => (out.push(String(s)), true)) as any);
  vi.spyOn(process.stderr, "write").mockImplementation(((s: any) => (err.push(String(s)), true)) as any);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  delete process.env.FREELM_CACHE_DIR;
  delete process.env.OPENROUTER_API_KEY;
});

it("--version prints the version", async () => {
  expect(await main(["--version"])).toBe(0);
  expect(out.join("")).toMatch(/freelm \d+\.\d+\.\d+/);
});

it("no args prints help", async () => {
  expect(await main([])).toBe(0);
  expect(out.join("")).toContain("chat");
});

it("no keys is a clean config error", async () => {
  expect(await main(["health"])).toBe(2);
  expect(err.join("")).toContain("config error");
});

it("chat prints the reply (provider note on stderr)", async () => {
  process.env.OPENROUTER_API_KEY = "sk-or-test";
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: any) =>
      String(url).includes("/models") ? new Response("nope", { status: 500 }) : new Response(OK, { status: 200 }),
    ),
  );
  expect(await main(["chat", "ping"])).toBe(0);
  expect(out.join("")).toContain("pong");
  expect(err.join("")).toContain("openrouter");
});

it("models lists the fallback catalog", async () => {
  process.env.OPENROUTER_API_KEY = "sk-or-test";
  vi.stubGlobal("fetch", vi.fn(async () => new Response("nope", { status: 500 })));
  expect(await main(["models", "--provider", "openrouter"])).toBe(0);
  const text = out.join("");
  expect(text).toContain("openrouter:");
  expect(text).toContain(":free");
});
