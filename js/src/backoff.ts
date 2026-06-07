/** Exponential backoff with full jitter. Delay in seconds. */
export function computeDelay(attempt: number, base = 0.5, factor = 2.0, cap = 30.0, jitter = true): number {
  attempt = Math.max(1, attempt);
  const raw = Math.min(cap, base * Math.pow(factor, attempt - 1));
  return jitter ? Math.random() * raw : raw;
}
