/** Monotonic seconds — for breaker/bucket/cooldown timing (testable, injectable). */
export function nowS(): number {
  return performance.now() / 1000;
}

/** Wall-clock seconds — for disk cache TTL. */
export function wallS(): number {
  return Date.now() / 1000;
}

/** The error to throw for an aborted signal (its reason, or a DOM AbortError). */
export function abortError(signal: AbortSignal): Error {
  const r = (signal as any).reason;
  return r instanceof Error ? r : new DOMException("This operation was aborted", "AbortError");
}

/** Sleep `ms`; rejects early with the abort reason if `signal` fires. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError(signal));
      return;
    }
    const onAbort = () => {
      clearTimeout(t);
      reject(abortError(signal!));
    };
    const t = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}
