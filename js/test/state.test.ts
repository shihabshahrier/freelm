import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import { FreeLLM, OpenRouter, StateStore } from "../src/index.js";

const OK = JSON.stringify({
  id: "x",
  model: "m",
  choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
});

afterEach(() => {
  vi.unstubAllGlobals();
  delete process.env.FREELM_CACHE_DIR;
});

it("roundtrips quota/cooldown/disable across 'restarts'", () => {
  const dir = mkdtempSync(join(tmpdir(), "freelm-state-"));
  const store = new StateStore(join(dir, "state.json"));
  const p = new OpenRouter("sk-or-abc", { discover: false });
  const k = p.keys[0];
  const now = 100;
  k.rpdUsed = 7;
  k.rpdReset = now + 1000;
  k.cooldownUntil = now + 30;
  k.disabled = true;
  k.lastError = "auth:401";
  store.save([p], now);

  const p2 = new OpenRouter("sk-or-abc", { discover: false });
  store.loadInto([p2], 5); // fresh process: different monotonic origin
  const k2 = p2.keys[0];
  expect(k2.rpdUsed).toBe(7);
  expect(k2.disabled).toBe(true);
  expect(k2.lastError).toBe("auth:401");
  expect(k2.cooldownUntil).toBeGreaterThan(5);
  expect(k2.rpdReset).toBeGreaterThan(5);
});

it("never writes raw keys", () => {
  const dir = mkdtempSync(join(tmpdir(), "freelm-state-"));
  const store = new StateStore(join(dir, "state.json"));
  store.save([new OpenRouter("sk-or-supersecret-key", { discover: false })], 0);
  expect(readFileSync(join(dir, "state.json"), "utf-8")).not.toContain("supersecret");
});

it("ignores a corrupt file", () => {
  const dir = mkdtempSync(join(tmpdir(), "freelm-state-"));
  const f = join(dir, "state.json");
  writeFileSync(f, "{not json");
  const store = new StateStore(f);
  const p = new OpenRouter("k", { discover: false });
  store.loadInto([p], 0); // must not throw
  expect(p.keys[0].rpdUsed).toBe(0);
  store.save([p], 0);
  expect(() => JSON.parse(readFileSync(f, "utf-8"))).not.toThrow();
});

it("client persist=true survives a restart", async () => {
  process.env.FREELM_CACHE_DIR = mkdtempSync(join(tmpdir(), "freelm-state-"));
  vi.stubGlobal("fetch", vi.fn(async () => new Response(OK, { status: 200 })));
  const llm = new FreeLLM([new OpenRouter("sk-or-x", { discover: false })], { persist: true });
  await llm.chat("hello");
  const used = llm.providers[0].keys[0].rpdUsed;
  expect(used).toBe(1);

  const llm2 = new FreeLLM([new OpenRouter("sk-or-x", { discover: false })], { persist: true });
  expect(llm2.providers[0].keys[0].rpdUsed).toBe(used);
});
