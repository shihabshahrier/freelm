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

it("free guard blocks paid passthrough; :free and optout pass", async () => {
  const { ConfigError } = await import("../src/errors.js");
  mockFetch(async () => new Response(OK("ok"), { status: 200 }));
  const llm = new FreeLLM([new OpenRouter("k", { discover: false })]);
  await expect(llm.chat("hi", { model: "openai/gpt-4o" })).rejects.toBeInstanceOf(ConfigError);
  const r = await llm.chat("hi", { model: "meta-llama/llama-3.3-70b-instruct:free" });
  expect(r.text).toBe("ok");
  const loose = new FreeLLM([new OpenRouter("k", { discover: false, freeOnly: false })]);
  expect((await loose.chat("hi", { model: "openai/gpt-4o" })).text).toBe("ok");
});

it("per-call model chain fails over to the second id", async () => {
  const bodies: any[] = [];
  let n = 0;
  mockFetch(async (_url, init) => {
    bodies.push(JSON.parse(init.body));
    n++;
    return n === 1
      ? new Response("model X is temporarily rate-limited upstream", { status: 429 })
      : new Response(OK("via-second"), { status: 200 });
  });
  const llm = new FreeLLM([new OpenRouter("k", { discover: false })]);
  const r = await llm.chat("hi", { model: ["first/model:free", "second/model:free"] });
  expect(r.text).toBe("via-second");
  expect(bodies[0].model).toBe("first/model:free");
  expect(bodies[1].model).toBe("second/model:free");
});

it("onEvent sees the failover sequence and survives a bad callback", async () => {
  let n = 0;
  mockFetch(async () => {
    n++;
    return n === 1 ? new Response("limit", { status: 429 }) : new Response(OK("done"), { status: 200 });
  });
  const events: string[] = [];
  const llm = new FreeLLM([new OpenRouter(["key-a", "key-b"], { discover: false })], {
    onEvent: (e) => {
      events.push(e.kind);
      throw new Error("callbacks must never break the call");
    },
  });
  expect((await llm.chat("hi")).text).toBe("done");
  expect(events).toEqual(["attempt", "error", "attempt", "success"]);
});

it("tools/response_format reach the body; toolCalls surfaced", async () => {
  const payload = JSON.parse(OK("calling"));
  payload.choices[0].message.tool_calls = [{ id: "call_1", type: "function", function: { name: "get_weather", arguments: "{}" } }];
  let body: any;
  mockFetch(async (_url, init) => {
    body = JSON.parse(init.body);
    return new Response(JSON.stringify(payload), { status: 200 });
  });
  const llm = new FreeLLM([new OpenRouter("k", { discover: false })]);
  const r = await llm.chat("weather?", {
    tools: [{ type: "function", function: { name: "get_weather", parameters: {} } }],
    tool_choice: "auto",
    response_format: { type: "json_object" },
  });
  expect(body.tools).toBeTruthy();
  expect(body.tool_choice).toBe("auto");
  expect(body.response_format).toEqual({ type: "json_object" });
  expect(r.toolCalls?.[0].function.name).toBe("get_weather");
});

it("quota 402 disables the key and fails over", async () => {
  mockFetch(async (url) =>
    url.includes("openrouter")
      ? new Response("Insufficient credits", { status: 402 })
      : new Response(OK("from-google"), { status: 200 }),
  );
  const llm = new FreeLLM([new OpenRouter("broke", { discover: false }), new GoogleAIStudio("k2")], {
    strategy: "priority",
  });
  const r = await llm.chat("hello");
  expect(r.provider).toBe("google");
  expect(llm.providers[0].keys[0].disabled).toBe(true);
  expect(llm.providers[0].keys[0].lastError).toBe("quota:402");
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
