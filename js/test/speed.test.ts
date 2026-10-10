// Failover speed: hedged attempts and smart (latency-aware) routing.
import { afterEach, expect, it, vi } from "vitest";
import * as engine from "../src/engine.js";
import { FreeLLM, modelSpec, NoProvidersAvailable, Provider } from "../src/index.js";
import { LATENCY_PRIOR_MS, LATENCY_TTL } from "../src/providers/base.js";
import { orderCandidates } from "../src/strategy.js";
import { collect, mockFetch, OK, sse } from "./helpers.js";

afterEach(() => {
  vi.unstubAllGlobals();
});

const prov = (name: string, opts: Record<string, any> = {}) =>
  new Provider(`k-${name}`, { name, baseUrl: `https://${name}.test/v1`, models: [modelSpec("m", ["chat"])], rpm: null, ...opts });

const SSE_OK = 'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n';

/** A response after `ms`, or an AbortError as soon as the request is aborted. */
function later(ms: number, init: any, res: () => Response, aborted?: () => void): Promise<Response> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => resolve(res()), ms);
    init?.signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        aborted?.();
        reject(new DOMException("aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

// -- pure helpers ---------------------------------------------------------------------

it("hedge delay adapts to measured latency", () => {
  const p = prov("a");
  const cand = { provider: p, key: p.keys[0], model: "m" };
  expect(engine.hedgeDelay(cand, false, false)).toBeNull();
  expect(engine.hedgeDelay(cand, 0, false)).toBeNull();
  expect(engine.hedgeDelay(cand, 2.5, false)).toBe(2.5); // fixed
  const [floor, cap, unmeasured] = engine.HEDGE_CHAT;
  expect(engine.hedgeDelay(cand, true, false)).toBe(unmeasured);
  expect(engine.hedgeDelay(cand, true, true)).toBe(engine.HEDGE_STREAM[2]); // streams hedge sooner
  p.keys[0].ewmaLatency = 100; // fast key: never below the floor
  expect(engine.hedgeDelay(cand, true, false)).toBe(floor);
  p.keys[0].ewmaLatency = 2000; // 3x its usual latency
  expect(engine.hedgeDelay(cand, true, false)).toBe(6);
  p.keys[0].ewmaLatency = 60000;
  expect(engine.hedgeDelay(cand, true, false)).toBe(cap);
});

it("smart order prefers fast, tries unknown, forgets old samples", () => {
  const fast = prov("fast");
  const slow = prov("slow");
  const fresh = prov("new");
  const now = 1000;
  fast.keys[0].ewmaLatency = 300;
  fast.keys[0].latencyAt = now;
  slow.keys[0].ewmaLatency = 9000;
  slow.keys[0].latencyAt = now;
  const names = () => orderCandidates([slow, fresh, fast], "auto", now, "smart", { p: 0 }).map((c) => c.provider.name);
  expect(names()).toEqual(["fast", "new", "slow"]); // unknown = typical (LATENCY_PRIOR_MS)
  expect(fresh.expectedLatency(now)).toBe(LATENCY_PRIOR_MS);
  slow.keys[0].latencyAt = now - LATENCY_TTL - 1; // stale: slow gets another chance
  expect(names()).toEqual(["fast", "slow", "new"]); // ties keep the given order
  fast.priority = 1; // priority tiers still come first
  expect(names().at(-1)).toBe("fast");
});

// -- the race ---------------------------------------------------------------------------

it("a slow provider is hedged, aborted and remembered", async () => {
  let aborted = false;
  const calls = mockFetch((url, _body, init) =>
    url.startsWith("https://a.test")
      ? later(5000, init, () => new Response(OK("slow")), () => (aborted = true))
      : new Response(OK("fast")),
  );
  const a = prov("a");
  const events: string[] = [];
  const llm = new FreeLLM([a, prov("b")], { hedge: 0.05, onEvent: (e) => events.push(`${e.kind}:${e.provider}`) });
  const t0 = performance.now();
  expect((await llm.chat("hi")).text).toBe("fast");
  expect(performance.now() - t0).toBeLessThan(500); // didn't wait for the slow one
  expect(events.slice(0, 2)).toEqual(["attempt:a", "hedge:b"]);
  expect(aborted).toBe(true);
  expect(a.keys[0].ewmaLatency).toBeGreaterThanOrEqual(50); // marked slow ...
  events.length = 0;
  expect((await llm.chat("again")).text).toBe("fast"); // ... so smart routing goes to b first
  expect(events.filter((e) => e.startsWith("attempt"))).toEqual(["attempt:b"]);
  expect(calls.length).toBe(3);
});

it("hedging off waits for the running attempt", async () => {
  const calls = mockFetch((url, _body, init) =>
    url.startsWith("https://a.test") ? later(200, init, () => new Response(OK("slow"))) : new Response(OK("fast")),
  );
  const events: string[] = [];
  const llm = new FreeLLM([prov("a"), prov("b")], { hedge: false, onEvent: (e) => events.push(e.kind) });
  expect((await llm.chat("hi")).text).toBe("slow");
  expect(calls.length).toBe(1);
  expect(events).not.toContain("hedge");
});

it("a hedge failing fast keeps waiting for the original", async () => {
  mockFetch((url, _body, init) =>
    url.startsWith("https://a.test")
      ? later(300, init, () => new Response(OK("slow")))
      : new Response("limit", { status: 429 }),
  );
  const llm = new FreeLLM([prov("a"), prov("b")], { hedge: 0.05 });
  expect((await llm.chat("hi")).text).toBe("slow");
});

it("a stalled stream is hedged before the first token", async () => {
  mockFetch((url, _body, init) =>
    url.startsWith("https://a.test") ? later(5000, init, () => sse(SSE_OK, "data: [DONE]\n\n")) : sse(SSE_OK, "data: [DONE]\n\n"),
  );
  const llm = new FreeLLM([prov("a"), prov("b")], { hedge: 0.05 });
  const t0 = performance.now();
  expect((await collect(llm.stream("hi"))).join("")).toBe("hi");
  expect(performance.now() - t0).toBeLessThan(500);
  const chunks = await collect(llm.streamChunks("hi"));
  expect(chunks.map((c) => c.choices[0].delta.content)).toEqual(["hi"]);
});

it("an attempt that outlives the deadline is recorded as a timeout", async () => {
  mockFetch((_url, _body, init) => later(5000, init, () => new Response(OK())));
  const a = prov("a");
  const llm = new FreeLLM([a], { timeout: 0.2 });
  const t0 = performance.now();
  await expect(llm.chat("hi")).rejects.toBeInstanceOf(NoProvidersAvailable);
  expect(performance.now() - t0).toBeLessThan(600);
  expect(a.keys[0].lastError).toBe("transient:0"); // cooled: the next call skips it
});

it("aborting the call aborts every running attempt", async () => {
  let abortedAttempts = 0;
  mockFetch((_url, _body, init) => later(5000, init, () => new Response(OK()), () => abortedAttempts++));
  const ac = new AbortController();
  const llm = new FreeLLM([prov("a"), prov("b")], { hedge: 0.05 });
  setTimeout(() => ac.abort(), 150); // after the hedge started
  await expect(llm.chat("hi", { signal: ac.signal })).rejects.toMatchObject({ name: "AbortError" });
  expect(abortedAttempts).toBe(2);
});
