/** Token bucket for requests-per-minute pacing. Single-threaded — no lock needed. */
export class TokenBucket {
  capacity: number;
  tokens: number;
  updated = 0;

  constructor(public ratePerMin: number, capacity?: number) {
    this.capacity = capacity ?? Math.max(1, ratePerMin);
    this.tokens = this.capacity;
  }

  private refill(now: number): void {
    if (this.updated === 0) {
      this.updated = now;
      return;
    }
    const dt = now - this.updated;
    if (dt <= 0) return;
    this.tokens = Math.min(this.capacity, this.tokens + dt * (this.ratePerMin / 60));
    this.updated = now;
  }

  peek(now: number): number {
    this.refill(now);
    return this.tokens;
  }

  consume(n: number, now: number): boolean {
    this.refill(now);
    if (this.tokens >= n) {
      this.tokens -= n;
      return true;
    }
    return false;
  }

  timeUntil(n: number, now: number): number {
    this.refill(now);
    if (this.tokens >= n) return 0;
    return (n - this.tokens) / (this.ratePerMin / 60);
  }
}
