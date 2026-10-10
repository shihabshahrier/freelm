/** Per-key runtime state: breaker + rpm bucket + daily quota + cooldowns. */
import { CircuitBreaker } from "./breaker.js";
import { TokenBucket } from "./ratelimit.js";

export const DAY = 86400.0;
/** Placeholder "key" for keyless providers (no Authorization header is sent). */
export const ANONYMOUS = "anonymous";
/** Stand-in for "unlimited" daily quota so it ranks high but stays finite/comparable. */
export const UNLIMITED = 100_000.0;

export class KeyState {
  breaker = new CircuitBreaker();
  bucket: TokenBucket | null = null;
  rpd: number | null = null;
  rpdUsed = 0;
  rpdReset = 0;
  cooldownUntil = 0;
  disabled = false;
  disabledSinceWall = 0; // wall-clock ts it was disabled (persistence TTL)
  ewmaLatency = 0;
  latencyAt = 0; // monotonic ts of the last latency sample
  lastError: string | null = null;
  /** model id -> monotonic ts until which this key must not use that model
   * (its own per-model quota hit, no access to the model, ...). */
  modelUntil = new Map<string, number>();

  constructor(public key: string, public tier = "free") {}

  private rollDaily(now: number): void {
    if (this.rpd === null) return;
    if (this.rpdReset === 0) this.rpdReset = now + DAY;
    else if (now >= this.rpdReset) {
      this.rpdUsed = 0;
      this.rpdReset = now + DAY;
    }
  }

  ready(now: number): boolean {
    if (this.disabled) return false;
    if (now < this.cooldownUntil) return false;
    if (!this.breaker.allow(now)) return false;
    this.rollDaily(now);
    if (this.rpd !== null && this.rpdUsed >= this.rpd) return false;
    if (this.bucket && this.bucket.peek(now) < 1) return false;
    return true;
  }

  reserve(now: number): boolean {
    this.rollDaily(now);
    if (this.bucket && !this.bucket.consume(1, now)) return false;
    this.rpdUsed++;
    return true;
  }

  remaining(now: number): number {
    if (!this.ready(now)) return 0;
    const daily = this.rpd === null ? UNLIMITED : Math.max(0, this.rpd - this.rpdUsed);
    const burst = this.bucket ? this.bucket.peek(now) : UNLIMITED;
    return Math.min(daily, burst);
  }

  waitTime(now: number): number | null {
    if (this.disabled) return null;
    const waits: number[] = [];
    if (now < this.cooldownUntil) waits.push(this.cooldownUntil - now);
    waits.push(this.breaker.timeUntilHalfOpen(now));
    this.rollDaily(now);
    if (this.rpd !== null && this.rpdUsed >= this.rpd) waits.push(Math.max(0, this.rpdReset - now));
    if (this.bucket && this.bucket.peek(now) < 1) waits.push(this.bucket.timeUntil(1, now));
    return waits.length ? Math.max(...waits) : 0;
  }

  benchModel(model: string, until: number): void {
    this.modelUntil.set(model, Math.max(until, this.modelUntil.get(model) ?? 0));
  }

  modelWait(model: string, now: number): number {
    const until = this.modelUntil.get(model);
    if (until === undefined) return 0;
    if (now >= until) {
      this.modelUntil.delete(model);
      return 0;
    }
    return until - now;
  }

  modelReady(model: string, now: number): boolean {
    return this.modelWait(model, now) === 0;
  }

  masked(): string {
    return maskKey(this.key);
  }

  /** Never put the raw key in logs: console.log / util.inspect / JSON.stringify
   * of a KeyState (or of an error's attempts) show the masked key only. */
  [Symbol.for("nodejs.util.inspect.custom")](): string {
    return `KeyState { key: '${this.masked()}', tier: '${this.tier}', disabled: ${this.disabled}, rpdUsed: ${this.rpdUsed}, lastError: ${JSON.stringify(this.lastError)} }`;
  }

  toJSON(): Record<string, unknown> {
    return { key: this.masked(), tier: this.tier, disabled: this.disabled, rpdUsed: this.rpdUsed, lastError: this.lastError };
  }
}

/** How a key is shown anywhere (events, health, errors, doctor) — never raw. */
export function maskKey(key: string): string {
  if (key === ANONYMOUS) return "(keyless)";
  return key.length > 12 ? `${key.slice(0, 6)}...${key.slice(-4)}` : "***";
}

export function newKeyState(key: string, tier: string, rpm: number | null, rpd: number | null): KeyState {
  const ks = new KeyState(key, tier);
  ks.bucket = rpm ? new TokenBucket(rpm) : null;
  ks.rpd = rpd;
  return ks;
}
