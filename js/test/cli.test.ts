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
  "KILO_API_KEY", "FREELM_KILO_KEYS",
];

let out: string[];
let err: string[];

beforeEach(() => {
  process.env.FREELM_CACHE_DIR = mkdtempSync(join(tmpdir(), "freelm-cli-"));
  for (const v of KEY_VARS) delete process.env[v];
  process.env.FREELM_KEYLESS = "0"; // tests opt in to keyless explicitly
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
  delete process.env.FREELM_KEYLESS;
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

it("doctor without keys prints signup links (exit 2)", async () => {
  expect(await main(["doctor"])).toBe(2);
  const text = out.join("");
  expect(text).toContain("GEMINI_API_KEY");
  expect(text).toContain("https://aistudio.google.com/apikey");
});

it("doctor reports each key with a fix, masked", async () => {
  process.env.OPENROUTER_API_KEY = "sk-or-dead-key-123456";
  process.env.GEMINI_API_KEY = "AIza-good-key-123456";
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: any) => {
      const u = String(url);
      if (u.endsWith("/models")) return new Response("nope", { status: 500 });
      if (u.includes("openrouter")) return new Response(JSON.stringify({ error: { message: "User not found.", code: 401 } }), { status: 401 });
      return new Response(OK, { status: 200 });
    }),
  );
  expect(await main(["doctor"])).toBe(0);
  const text = out.join("");
  expect(text).toContain("FAIL");
  expect(text).toContain("User not found.");
  expect(text).toContain("https://openrouter.ai/keys");
  expect(text).toContain("1 of 2 key(s) working");
  expect(text).not.toContain("dead-key");
  delete process.env.GEMINI_API_KEY;
});

it("doctor --json; all failing exits 1", async () => {
  process.env.OPENROUTER_API_KEY = "sk-or-dead-key-123456";
  vi.stubGlobal("fetch", vi.fn(async (url: any) => (String(url).endsWith("/models") ? new Response("", { status: 500 }) : new Response("Payment required", { status: 402 }))));
  expect(await main(["doctor", "--json"])).toBe(1);
  const data = JSON.parse(out.join(""));
  expect(data.keys[0].status).toBe("FAIL");
  expect(data.missing.some((m: any) => m.provider === "groq")).toBe(true);
});

it("parses --model=x, rejects a missing value, and --version inside a prompt is just text", async () => {
  process.env.OPENROUTER_API_KEY = "sk-or-test";
  const bodies: any[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: any, init: any) => {
      if (String(url).endsWith("/models")) return new Response("", { status: 500 });
      bodies.push(JSON.parse(init.body));
      return new Response(OK, { status: 200 });
    }),
  );
  expect(await main(["chat", "what does --version do", "--model=fast"])).toBe(0);
  expect(bodies[0].messages[0].content).toBe("what does --version do");
  expect(await main(["chat", "hi", "--model"])).toBe(2);
  expect(err.join("")).toContain("usage error");
});

it("chat without keys uses the keyless endpoints, with a notice and no credentials", async () => {
  delete process.env.FREELM_KEYLESS; // CLI default: auto
  const auth: Array<string | undefined> = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: any, init: any) => {
      const u = String(url);
      if (u.endsWith("/models")) return new Response("", { status: 500 });
      if (u.includes("kilo.ai")) {
        auth.push(init.headers.Authorization);
        return new Response(OK, { status: 200 });
      }
      return new Response("limit", { status: 429 });
    }),
  );
  expect(await main(["chat", "ping"])).toBe(0);
  expect(out.join("")).toContain("pong");
  expect(err.join("")).toContain("keyless public endpoints");
  expect(auth).toEqual([undefined]);
});

it("doctor without keys checks the keyless endpoints", async () => {
  delete process.env.FREELM_KEYLESS;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: any) => {
      const u = String(url);
      if (u.endsWith("/models")) return new Response("", { status: 500 });
      if (u.includes("kilo.ai")) return new Response(OK, { status: 200 });
      return new Response("limit", { status: 429, headers: { "retry-after": "30" } });
    }),
  );
  expect(await main(["doctor"])).toBe(0);
  const text = out.join("");
  expect(text).toContain("GEMINI_API_KEY");
  expect(text).toContain("(keyless)");
  expect(text).toContain("no keys configured, 2 keyless endpoint(s) up — ready: kilo, ovh");
});
