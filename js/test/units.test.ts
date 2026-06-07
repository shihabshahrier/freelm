import { describe, expect, it } from "vitest";
import { TokenBucket } from "../src/ratelimit.js";
import { CircuitBreaker } from "../src/breaker.js";
import { newKeyState } from "../src/keys.js";
import { modelSpec, resolveModels } from "../src/index.js";

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
});
