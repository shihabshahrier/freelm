/** Exception hierarchy + HTTP-status -> error classification. */

export class FreeLLMError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

export class ConfigError extends FreeLLMError {}

export type RateScope = "key" | "model";

export class ProviderError extends FreeLLMError {
  constructor(
    public provider: string,
    public status: number,
    public detail = "",
    public retryable = false,
    public retryAfter: number | null = null,
    public modelMissing = false,
    public scope: RateScope = "key",
  ) {
    super(`[${provider}] ${status} ${detail}`.trim());
  }
}

export class AuthError extends ProviderError {
  constructor(provider: string, status: number, detail = "") {
    super(provider, status, detail, false);
  }
}

export class RateLimited extends ProviderError {
  constructor(provider: string, status: number, detail = "", retryAfter: number | null = null, scope: RateScope = "key") {
    super(provider, status, detail, true, retryAfter, false, scope);
  }
}

export class Transient extends ProviderError {
  constructor(provider: string, status: number, detail = "", retryAfter: number | null = null) {
    super(provider, status, detail, true, retryAfter);
  }
}

export class ModelNotFound extends ProviderError {
  constructor(provider: string, status: number, detail = "") {
    super(provider, status, detail, false, null, true);
  }
}

export class NoProvidersAvailable extends FreeLLMError {
  constructor(public attempts: Array<[any, Error]>) {
    const detail = attempts
      .slice(0, 8)
      .map(([c, e]) => `${c.provider.name}/${c.key.masked()}:${e.constructor.name}`)
      .join("; ");
    super(`all providers/keys exhausted after ${attempts.length} attempt(s): ${detail || "none ready"}`);
  }
}

export function parseRetryAfter(value: string | null | undefined): number | null {
  if (!value) return null;
  const n = Number(value);
  if (!Number.isNaN(n)) return Math.max(0, n);
  const t = Date.parse(value);
  if (!Number.isNaN(t)) return Math.max(0, (t - Date.now()) / 1000);
  return null;
}

const TRANSIENT_STATUS = new Set([408, 409, 425, 500, 502, 503, 504, 529]);

export function classify(status: number, headers: Record<string, string>, body: string, provider: string): ProviderError {
  const retryAfter = parseRetryAfter(headers?.["retry-after"]);
  const msg = (body || "").slice(0, 300);
  const low = (body || "").toLowerCase();
  if (status === 401 || status === 403) return new AuthError(provider, status, msg);
  if (status === 429) return new RateLimited(provider, status, msg, retryAfter);
  if (TRANSIENT_STATUS.has(status)) return new Transient(provider, status, msg, retryAfter);
  if (status === 404 || ((status === 400 || status === 422) && low.includes("model"))) {
    return new ModelNotFound(provider, status, msg);
  }
  return new ProviderError(provider, status, msg, false);
}
