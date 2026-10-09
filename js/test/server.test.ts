// `freelm serve`: a real localhost socket; upstream fetch is stubbed, while
// the test talks to the server through node:http (unaffected by the stub).
import { request } from "node:http";
import type { Server } from "node:http";
import { afterEach, expect, it, vi } from "vitest";
import { FreeLLM, GoogleAIStudio, OpenRouter } from "../src/index.js";
import { createServer } from "../src/server.js";
import { OK } from "./helpers.js";

const servers: Server[] = [];
afterEach(async () => {
  vi.unstubAllGlobals();
  await Promise.all(servers.splice(0).map((s) => new Promise((r) => s.close(r))));
});

async function start(llm: FreeLLM, opts = {}): Promise<number> {
  const s = createServer(llm, opts);
  servers.push(s);
  await new Promise<void>((r) => s.listen(0, "127.0.0.1", () => r()));
  return (s.address() as any).port;
}

function call(port: number, method: string, path: string, body?: any, headers: Record<string, string> = {}) {
  return new Promise<{ status: number; headers: any; text: string }>((resolve, reject) => {
    const data = body === undefined ? undefined : JSON.stringify(body);
    const req = request(
      { host: "127.0.0.1", port, method, path, headers: { "Content-Type": "application/json", ...(data ? { "Content-Length": Buffer.byteLength(data) } : {}), ...headers } },
      (res) => {
        let text = "";
        res.on("data", (c) => (text += c));
        res.on("end", () => resolve({ status: res.statusCode!, headers: res.headers, text }));
      },
    );
    req.on("error", reject);
    if (data) req.write(data);
    req.end();
  });
}

const msgs = [{ role: "user", content: "hi" }];

it("serves JSON chat completions with failover and provider headers", async () => {
  vi.stubGlobal("fetch", vi.fn(async (u: string) => (u.includes("openrouter") ? new Response("slow", { status: 429 }) : new Response(OK("pong", "gemini-x"), { status: 200 }))));
  const port = await start(new FreeLLM([new OpenRouter("k", { discover: false }), new GoogleAIStudio("k2")]));
  const r = await call(port, "POST", "/v1/chat/completions", { model: "auto", messages: msgs });
  expect(r.status).toBe(200);
  const body = JSON.parse(r.text);
  expect(body.object).toBe("chat.completion");
  expect(body.choices[0].message.content).toBe("pong");
  expect(r.headers["x-freellm-provider"]).toBe("google");
});

it("streams SSE chunks ending with [DONE]", async () => {
  const sse =
    'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n' +
    'data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n' +
    'data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n' +
    "data: [DONE]\n\n";
  vi.stubGlobal("fetch", vi.fn(async () => new Response(sse, { status: 200 })));
  const port = await start(new FreeLLM([new OpenRouter("k", { discover: false })]));
  const r = await call(port, "POST", "/v1/chat/completions", { model: "auto", stream: true, messages: msgs });
  expect(r.status).toBe(200);
  expect(r.headers["content-type"]).toBe("text/event-stream");
  const events = r.text.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6));
  expect(events[events.length - 1]).toBe("[DONE]");
  const chunks = events.slice(0, -1).map((e) => JSON.parse(e));
  expect(chunks.map((c) => c.choices[0].delta.content ?? "").join("")).toBe("Hello");
  expect(new Set(chunks.map((c) => c.id)).size).toBe(1);
});

it("maps errors to OpenAI shapes; enforces the API key; health stays open", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response("bad key", { status: 401 })));
  const port = await start(new FreeLLM([new OpenRouter("k", { discover: false })]), { apiKey: "s3cret" });
  expect((await call(port, "POST", "/v1/chat/completions", { model: "auto", messages: msgs })).status).toBe(401);
  const r = await call(port, "POST", "/v1/chat/completions", { model: "auto", messages: msgs }, { Authorization: "Bearer s3cret" });
  expect(r.status).toBe(503);
  expect(JSON.parse(r.text).error.code).toBe("no_providers_available");
  expect((await call(port, "POST", "/v1/chat/completions", { model: "auto" }, { Authorization: "Bearer s3cret" })).status).toBe(400);
  expect((await call(port, "GET", "/health")).status).toBe(200);
  expect((await call(port, "GET", "/v1/models")).status).toBe(401);
});

it("unknown model ids fall back to auto; /v1/models lists aliases first", async () => {
  const seen: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_u: string, init: any) => {
      const m = JSON.parse(init.body).model;
      seen.push(m);
      return m === "gpt-4o" ? new Response("model not found", { status: 404 }) : new Response(OK("ok"), { status: 200 });
    }),
  );
  const port = await start(new FreeLLM([new GoogleAIStudio("k")]));
  const r = await call(port, "POST", "/v1/chat/completions", { model: "gpt-4o", messages: msgs });
  expect(r.status).toBe(200);
  expect(seen[0]).toBe("gpt-4o");
  const models = JSON.parse((await call(port, "GET", "/v1/models")).text);
  expect(models.data[0].id).toBe("auto");
});
