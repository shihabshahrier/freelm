// Mirrors tests/test_review_fixes.py: regressions from the adversarial review of 0.4.
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { request } from "node:http";
import type { Server } from "node:http";
import { createServer as netServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { keyId } from "../src/state.js";
import {
  BadRequest,
  FreeLLM,
  GoogleAIStudio,
  Groq,
  NoProvidersAvailable,
  OpenRouter,
  StateStore,
  modelSpec,
} from "../src/index.js";
import { createServer } from "../src/server.js";
import { OK, collect, mockFetch, sse } from "./helpers.js";

const servers: Server[] = [];
afterEach(async () => {
  vi.unstubAllGlobals();
  await Promise.all(servers.splice(0).map((s) => new Promise((r) => s.close(r))));
});

const keyOf = (init: any) => String(init.headers.Authorization).split(" ").pop();

describe("benches caused by one key are per (key, model)", () => {
  it("a model-scoped 429 on one key leaves the other keys usable", async () => {
    const keys: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_u: string, init: any) => {
        keys.push(keyOf(init)!);
        return keyOf(init) === "key1" ? new Response("Rate limit reached for model qwen", { status: 429 }) : new Response(OK("from-key2"), { status: 200 });
      }) as any,
    );
    const llm = new FreeLLM([new Groq(["key1", "key2"], { models: [modelSpec("qwen", ["chat"])] })]);
    expect((await llm.chat("one")).text).toBe("from-key2");
    expect((await llm.chat("two")).text).toBe("from-key2");
    expect(keys).toEqual(["key1", "key2", "key2"]);
  });

  it("an access 404 benches only that key; a retired 410 benches the model", async () => {
    const noAccess = '{"error":{"message":"The model `qwen` does not exist or you do not have access to it."}}';
    const keys: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_u: string, init: any) => {
        keys.push(keyOf(init)!);
        return keyOf(init) === "key1" ? new Response(noAccess, { status: 404 }) : new Response(OK("ok"), { status: 200 });
      }) as any,
    );
    const groq = new Groq(["key1", "key2"], { models: [modelSpec("qwen", ["chat"])] });
    expect((await new FreeLLM([groq]).chat("hi")).text).toBe("ok");
    expect(keys).toEqual(["key1", "key2"]);
    expect(groq.modelReady("qwen", performance.now() / 1000)).toBe(true);
  });

  it("OpenRouter's upstream throttle benches the model on every key", async () => {
    const calls = mockFetch((_u, body) =>
      body.model === "m1:free" ? new Response("m1:free is temporarily rate-limited upstream", { status: 429 }) : new Response(OK("m2"), { status: 200 }),
    );
    const models = [modelSpec("m1:free", ["chat"]), modelSpec("m2:free", ["chat"])];
    expect((await new FreeLLM([new OpenRouter(["k1", "k2"], { models })]).chat("hi")).text).toBe("m2");
    expect(calls.map((c) => c.model)).toEqual(["m1:free", "m2:free"]);
  });
});

it("wait mode waits out a per-model quota", async () => {
  let n = 0;
  mockFetch(() => (++n === 1 ? new Response('{"error":{"details":[{"retryDelay":"1s"}]}}', { status: 429 }) : new Response(OK("after-wait"), { status: 200 })));
  const llm = new FreeLLM([new GoogleAIStudio("k", { models: [modelSpec("m1", ["chat"])] })], { wait: true, maxWait: 5, timeout: 10 });
  const t0 = performance.now();
  expect((await llm.chat("hi")).text).toBe("after-wait");
  const dt = performance.now() - t0;
  expect(dt).toBeGreaterThan(900);
  expect(dt).toBeLessThan(4000);
});

it("unicode line separators inside content survive streaming", async () => {
  mockFetch(() => sse('data: {"choices":[{"delta":{"content":"Hello world"}}]}\n\n', 'data: {"choices":[{"delta":{"content":"\u0085!"}}]}\n\n', "data: [DONE]\n\n"));
  expect((await collect(new FreeLLM([new OpenRouter("k", { discover: false })]).stream("hi"))).join("")).toBe("Hello world\u0085!");
});

it("an expired persisted disable is re-stamped when the key fails again", () => {
  const dir = mkdtempSync(join(tmpdir(), "freelm-restamp-"));
  const f = join(dir, "state.json");
  writeFileSync(f, JSON.stringify({ [keyId("openrouter", "k")!]: { disabled: true, disabled_since_wall: Date.now() / 1000 - 90000 } }));
  const store = new StateStore(f);
  const p = new OpenRouter("k", { discover: false });
  store.loadInto([p], 0);
  expect(p.keys[0].disabled).toBe(false);
  p.keys[0].disabled = true;
  store.save([p], 0);
  const saved = JSON.parse(readFileSync(f, "utf-8"))[keyId("openrouter", "k")!];
  expect(Date.now() / 1000 - saved.disabled_since_wall).toBeLessThan(60);
});

it("an empty or role-only 200 stream fails over", async () => {
  const calls = mockFetch((u) =>
    u.includes("openrouter") ? sse('data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n') : sse('data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', "data: [DONE]\n\n"),
  );
  expect((await collect(new FreeLLM([new OpenRouter("k", { discover: false }), new GoogleAIStudio("k2")]).stream("hi"))).join("")).toBe("ok");
  expect(calls.filter((c) => c.url.includes("googleapis")).length).toBe(1);
});

it("SSE `data: null` and list-valued deltas", async () => {
  mockFetch(() =>
    sse(
      "data: null\n",
      'data: {"choices":[{"delta":{"content":[{"type":"text","text":"A"},{"type":"image_url"}]}}]}\n',
      'data: {"choices":[{"delta":{"content":"B"}}]}\n',
      "data: [DONE]\n",
      'data: {"choices":[{"delta":{"content":"late"}}]}\n',
    ),
  );
  expect((await collect(new FreeLLM([new OpenRouter("k", { discover: false })]).stream("hi"))).join("")).toBe("AB");
});

it("one rejection while others cool is not reported as a caller bug", async () => {
  mockFetch(() => new Response("unsupported parameter: seed", { status: 400 }));
  const g = new GoogleAIStudio("k2", { models: [modelSpec("m1", ["chat"])] });
  g.keys[0].cooldownUntil = performance.now() / 1000 + 60;
  await expect(new FreeLLM([new OpenRouter("k", { discover: false }), g]).chat("hi", { seed: 1 })).rejects.toBeInstanceOf(NoProvidersAvailable);
  await expect(new FreeLLM([new OpenRouter("k", { discover: false })]).chat("hi", { seed: 1 })).rejects.toBeInstanceOf(BadRequest);
});

describe("server", () => {
  async function start(llm: FreeLLM, opts = {}): Promise<number> {
    const s = createServer(llm, opts);
    servers.push(s);
    await new Promise<void>((r) => s.listen(0, "127.0.0.1", () => r()));
    return (s.address() as any).port;
  }

  function raw(port: number, method: string, path: string, headers: Record<string, string>, body?: string) {
    return new Promise<{ status: number; headers: any; text: string }>((resolve, reject) => {
      const req = request({ host: "127.0.0.1", port, method, path, headers: { ...(body ? { "Content-Length": Buffer.byteLength(body) } : {}), ...headers } }, (res) => {
        let text = "";
        res.on("data", (c) => (text += c));
        res.on("end", () => resolve({ status: res.statusCode!, headers: res.headers, text }));
      });
      req.on("error", reject);
      if (body) req.write(body);
      req.end();
    });
  }

  it("refuses browser-style requests: text/plain POSTs and foreign Host headers", async () => {
    const port = await start(new FreeLLM([new OpenRouter("k", { discover: false })]));
    expect((await raw(port, "POST", "/v1/chat/completions", { "Content-Type": "text/plain" }, '{"messages":[]}')).status).toBe(415);
    expect((await raw(port, "GET", "/v1/models", { Host: "evil.example:4000" })).status).toBe(403);
  });

  it("CORS preflight reflects the requested headers", async () => {
    const port = await start(new FreeLLM([new OpenRouter("k", { discover: false })]), { cors: true });
    const r = await raw(port, "OPTIONS", "/v1/chat/completions", { "Access-Control-Request-Headers": "x-stainless-os, authorization" });
    expect(r.status).toBe(204);
    expect(r.headers["access-control-allow-headers"]).toBe("x-stainless-os, authorization");
  });

  it("an oversized body gets a 413 response, not a reset", async () => {
    const port = await start(new FreeLLM([new OpenRouter("k", { discover: false })]));
    const big = JSON.stringify({ messages: [{ role: "user", content: "x".repeat(21 * 1024 * 1024) }] });
    const r = await raw(port, "POST", "/v1/chat/completions", { "Content-Type": "application/json" }, big);
    expect(r.status).toBe(413);
  });

  it("/health uses the same snake_case fields as the Python server", async () => {
    const port = await start(new FreeLLM([new OpenRouter("k", { discover: false })]));
    const r = JSON.parse((await raw(port, "GET", "/health", {})).text);
    expect(Object.keys(r.keys[0])).toEqual(expect.arrayContaining(["rpd_used", "last_error", "ewma_latency_ms"]));
  });

  it("a busy port surfaces EADDRINUSE", async () => {
    const busy = netServer();
    await new Promise<void>((r) => busy.listen(0, "127.0.0.1", () => r()));
    const port = (busy.address() as any).port;
    const s = createServer(new FreeLLM([new OpenRouter("k", { discover: false })]));
    const err = await new Promise<any>((resolve) => {
      s.once("error", resolve);
      s.listen(port, "127.0.0.1");
    });
    expect(err.code).toBe("EADDRINUSE");
    await new Promise((r) => busy.close(r));
  });
});
