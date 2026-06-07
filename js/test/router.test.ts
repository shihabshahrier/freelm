import { afterEach, expect, it, vi } from "vitest";
import { FreeLLM, GoogleAIStudio, NoProvidersAvailable, OpenRouter, modelSpec } from "../src/index.js";

const OK = (content = "hi", model = "m") =>
  JSON.stringify({
    id: "x",
    model,
    choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  });

function mockFetch(handler: (url: string, init: any) => Promise<Response>) {
  vi.stubGlobal("fetch", vi.fn(handler as any));
}

afterEach(() => vi.unstubAllGlobals());

it("success path", async () => {
  mockFetch(async () => new Response(OK("hi"), { status: 200 }));
  const llm = new FreeLLM([new OpenRouter("k", { discover: false })]);
  const r = await llm.chat("hello");
  expect(r.text).toBe("hi");
  expect(r.provider).toBe("openrouter");
  expect(r.usage.total_tokens).toBe(5);
});

it("rotates key on 429", async () => {
  let n = 0;
  mockFetch(async () => {
    n++;
    return n === 1 ? new Response("account rate limit", { status: 429 }) : new Response(OK("second"), { status: 200 });
  });
  const llm = new FreeLLM([new OpenRouter(["key-a", "key-b"], { discover: false })]);
  const r = await llm.chat("hello");
  expect(r.text).toBe("second");
  expect(n).toBe(2);
});

it("fails over across providers", async () => {
  mockFetch(async (url) =>
    url.includes("openrouter") ? new Response("acct rate", { status: 429 }) : new Response(OK("from-google"), { status: 200 }),
  );
  const llm = new FreeLLM([new OpenRouter("k", { discover: false }), new GoogleAIStudio("k2")], { strategy: "priority" });
  const r = await llm.chat("hello");
  expect(r.provider).toBe("google");
});

it("interleave reaches provider 2 despite many throttled models", async () => {
  const many = Array.from({ length: 10 }, (_, i) => modelSpec(`vendor/m${i}:free`, ["chat", "large"]));
  mockFetch(async (url) =>
    url.includes("openrouter")
      ? new Response("temporarily rate-limited upstream", { status: 429 })
      : new Response(OK("google"), { status: 200 }),
  );
  const llm = new FreeLLM([new OpenRouter("k", { discover: false, models: many }), new GoogleAIStudio("k2")], {
    strategy: "priority",
  });
  const r = await llm.chat("hi");
  expect(r.provider).toBe("google");
});

it("model-scoped 429 tries next model on same key", async () => {
  let n = 0;
  mockFetch(async () => {
    n++;
    return n === 1
      ? new Response("model X is temporarily rate-limited upstream", { status: 429 })
      : new Response(OK("recovered"), { status: 200 });
  });
  const llm = new FreeLLM([new OpenRouter("only-key", { discover: false })]);
  const r = await llm.chat("hi");
  expect(r.text).toBe("recovered");
  expect(n).toBe(2);
  expect(llm.providers[0].keys[0].cooldownUntil).toBe(0);
});

it("auth disables key then exhausts", async () => {
  mockFetch(async () => new Response("invalid key", { status: 401 }));
  const llm = new FreeLLM([new OpenRouter("bad", { discover: false })]);
  await expect(llm.chat("hi")).rejects.toBeInstanceOf(NoProvidersAvailable);
  expect(llm.providers[0].keys[0].disabled).toBe(true);
});

it("bad request (400) raises immediately", async () => {
  mockFetch(async () => new Response("invalid temperature", { status: 400 }));
  const llm = new FreeLLM([new OpenRouter("k", { discover: false }), new GoogleAIStudio("k2")]);
  await expect(llm.chat("hi")).rejects.toThrow(/400/);
});

it("health report", async () => {
  mockFetch(async () => new Response(OK(), { status: 200 }));
  const llm = new FreeLLM([new OpenRouter("k", { discover: false })]);
  await llm.chat("hi");
  const h = llm.health();
  expect(h[0].provider).toBe("openrouter");
  expect(h[0].breaker).toBe("closed");
});
