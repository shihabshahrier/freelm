/** Monotonic seconds — for breaker/bucket/cooldown timing (testable, injectable). */
export function nowS(): number {
  return performance.now() / 1000;
}

/** Wall-clock seconds — for disk cache TTL. */
export function wallS(): number {
  return Date.now() / 1000;
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
