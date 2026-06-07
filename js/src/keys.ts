/** Per-key runtime state: breaker + rpm bucket + daily quota + cooldowns. */
import { CircuitBreaker } from "./breaker.js";
import { TokenBucket } from "./ratelimit.js";

export const DAY = 86400.0;
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
  ewmaLatency = 0;
  lastError: string | null = null;

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

  masked(): string {
    const k = this.key;
    return k.length > 12 ? `${k.slice(0, 6)}...${k.slice(-4)}` : "***";
  }
}

export function newKeyState(key: string, tier: string, rpm: number | null, rpd: number | null): KeyState {
  const ks = new KeyState(key, tier);
  ks.bucket = rpm ? new TokenBucket(rpm) : null;
  ks.rpd = rpd;
  return ks;
}
