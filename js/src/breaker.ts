/** Per-key circuit breaker. Time (monotonic seconds) is injected for testability. */
export type BreakerState = "closed" | "open" | "half_open";

export class CircuitBreaker {
  state: BreakerState = "closed";
  failures = 0;
  openedAt = 0;

  constructor(public failThreshold = 4, public cooldown = 30.0) {}

  allow(now: number): boolean {
    if (this.state === "open") {
      if (now - this.openedAt >= this.cooldown) {
        this.state = "half_open";
        return true;
      }
      return false;
    }
    return true;
  }

  onSuccess(): void {
    this.failures = 0;
    this.state = "closed";
  }

  onFailure(now: number): void {
    this.failures++;
    if (this.state === "half_open" || this.failures >= this.failThreshold) {
      this.state = "open";
      this.openedAt = now;
    }
  }

  timeUntilHalfOpen(now: number): number {
    if (this.state !== "open") return 0;
    return Math.max(0, this.cooldown - (now - this.openedAt));
  }
}
