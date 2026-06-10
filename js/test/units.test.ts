import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { TokenBucket } from "../src/ratelimit.js";
import { CircuitBreaker } from "../src/breaker.js";
import { newKeyState } from "../src/keys.js";
import { applySuccess } from "../src/engine.js";
import { modelSpec, resolveModels, VERSION } from "../src/index.js";

describe("TokenBucket", () => {
  it("consumes and refills", () => {
    const b = new TokenBucket(60, 2); // 1 token/sec, capacity 2
    const now = 100;
    expect(b.consume(1, now)).toBe(true);
    expect(b.consume(1, now)).toBe(true);
    expect(b.consume(1, now)).toBe(false);
    expect(Math.round(b.timeUntil(1, now) * 100) / 100).toBe(1);
    expect(b.consume(1, now + 1)).toBe(true);
  });
});

describe("CircuitBreaker", () => {
  it("opens then half-opens", () => {
    const cb = new CircuitBreaker(2, 10);
    expect(cb.allow(0)).toBe(true);
    cb.onFailure(0);
    expect(cb.state).toBe("closed");
    cb.onFailure(0);
    expect(cb.state).toBe("open");
    expect(cb.allow(0)).toBe(false);
    expect(cb.allow(11)).toBe(true);
    cb.onSuccess();
    expect(cb.state).toBe("closed");
  });
});

describe("KeyState", () => {
  it("daily quota + reset", () => {
    const k = newKeyState("k", "free", null, 2);
    const now = 1000;
    expect(k.ready(now)).toBe(true);
    expect(k.reserve(now)).toBe(true);
    expect(k.reserve(now)).toBe(true);
    expect(k.ready(now)).toBe(false);
    expect(k.ready(now + 86401)).toBe(true);
  });
});

describe("resolveModels", () => {
  it("resolves aliases and passthrough", () => {
    const models = [modelSpec("big/model", ["chat", "large"]), modelSpec("small/model", ["chat", "small", "fast"])];
    expect(resolveModels(models, "auto")).toEqual(["big/model", "small/model"]);
    expect(resolveModels(models, "chat:large")).toEqual(["big/model"]);
    expect(resolveModels(models, "fast")).toEqual(["small/model"]);
    expect(resolveModels(models, "small/model")).toEqual(["small/model"]);
    expect(resolveModels(models, "vendor/unknown-xyz")).toEqual(["vendor/unknown-xyz"]);
  });

  it("passes through unknown ids with a colon suffix verbatim", () => {
    // must never silently fan out to the whole chat list
    const models = [modelSpec("big/model:free", ["chat", "large"])];
    expect(resolveModels(models, "moonshotai/kimi-k2:free")).toEqual(["moonshotai/kimi-k2:free"]);
    expect(resolveModels(models, "big/model:free")).toEqual(["big/model:free"]); // exact still wins
  });

  it("orders by ModelSpec priority", () => {
    const models = [
      modelSpec("late/model", ["chat", "large"], 0, true, 5),
      modelSpec("early/model", ["chat", "large"], 0, true, 0),
      modelSpec("middle/model", ["chat", "large"], 0, true, 2),
    ];
    expect(resolveModels(models, "auto")).toEqual(["early/model", "middle/model", "late/model"]);
  });

  it("routes tag aliases (tools/vision) to tagged models", () => {
    const models = [
      modelSpec("plain/model", ["chat"]),
      modelSpec("tooly/model", ["chat", "tools"]),
      modelSpec("eyes/model", ["chat", "vision"]),
    ];
    expect(resolveModels(models, "chat:tools")).toEqual(["tooly/model"]);
    expect(resolveModels(models, "vision")).toEqual(["eyes/model"]);
    expect(resolveModels(models, "reasoning")).toEqual(["plain/model", "tooly/model", "eyes/model"]); // none tagged -> all chat
  });
});

describe("Provider resolveModels", () => {
  it("prefer= reorders resolved models but not direct asks", async () => {
    const { Provider } = await import("../src/providers/base.js");
    const p = new Provider("k", {
      name: "x",
      baseUrl: "https://x.test/v1",
      models: [
        modelSpec("a/first:free", ["chat"]),
        modelSpec("b/qwen3-80b:free", ["chat"]),
        modelSpec("c/last:free", ["chat"]),
      ],
      prefer: ["c/last:free", "qwen3"], // exact id, then substring
    });
    expect(p.resolveModels("auto")).toEqual(["c/last:free", "b/qwen3-80b:free", "a/first:free"]);
    expect(p.resolveModels("b/qwen3-80b:free")).toEqual(["b/qwen3-80b:free"]); // direct ask not reordered
  });

  it("resolves per-call chains in order, deduped", async () => {
    const { Provider } = await import("../src/providers/base.js");
    const p = new Provider("k", {
      name: "x",
      baseUrl: "https://x.test/v1",
      models: [modelSpec("big/model", ["chat", "large"]), modelSpec("small/model", ["chat", "small", "fast"])],
    });
    expect(p.resolveModels(["vendor/custom:free", "fast"])).toEqual(["vendor/custom:free", "small/model"]);
    expect(p.resolveModels(["fast", "auto"])).toEqual(["small/model", "big/model"]);
  });

  it("provider priority breaks ties in dynamic strategies", async () => {
    const { Provider } = await import("../src/providers/base.js");
    const { orderCandidates } = await import("../src/strategy.js");
    const mk = (name: string, prio: number) =>
      new Provider("k", { name, baseUrl: "https://x.test/v1", priority: prio, models: [modelSpec("m", ["chat"])] });
    for (const strat of ["quota_aware", "latency", "round_robin"]) {
      const cands = orderCandidates([mk("second", 1), mk("first", 0)], "auto", 0, strat, { p: 0 });
      expect(cands[0].provider.name, strat).toBe("first");
    }
  });
});

describe("applySuccess", () => {
  it("ignores zero-latency samples instead of decaying the EWMA", () => {
    const k = newKeyState("k", "free", null, null);
    k.ewmaLatency = 100;
    const cand = { provider: null, key: k, model: "m" };
    applySuccess(cand as any, 0); // "no sample" (e.g. empty stream)
    expect(k.ewmaLatency).toBe(100);
    applySuccess(cand as any, 200);
    expect(k.ewmaLatency).toBeCloseTo(100 * 0.7 + 200 * 0.3);
  });
});

describe("VERSION", () => {
  it("matches package.json", () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf-8"));
    expect(VERSION).toBe(pkg.version);
  });
});
