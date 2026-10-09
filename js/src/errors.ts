/** Exception hierarchy + HTTP-status -> error classification. */

export class FreeLLMError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

export class ConfigError extends FreeLLMError {}

/** "key": this key/account (cool it) · "model": this key's quota for one model
 * (bench the pair) · "upstream": the model for everyone (bench it on every key). */
export type ErrorScope = "key" | "model" | "upstream";
/** @deprecated alias kept for 0.2.x imports — use ErrorScope. */
export type RateScope = ErrorScope;

export class ProviderError extends FreeLLMError {
  constructor(
    public provider: string,
    public status: number,
    public detail = "",
    public retryable = false,
    public retryAfter: number | null = null,
    public modelMissing = false,
    public scope: ErrorScope = "key",
  ) {
    super(`[${provider}] ${status} ${detail}`.trim());
  }
}

/** 401/403 — key is invalid or lacks access. Key gets disabled. */
export class AuthError extends ProviderError {
  constructor(provider: string, status: number, detail = "") {
    super(provider, status, detail, false);
  }
}

/** 429 — rotate to another key/provider. `scope` "key" cools the key; "model"
 * benches only the throttled model (e.g. Gemini/Groq per-model quotas). */
export class RateLimited extends ProviderError {
  constructor(provider: string, status: number, detail = "", retryAfter: number | null = null, scope: ErrorScope = "key") {
    super(provider, status, detail, true, retryAfter, false, scope);
  }
}

/** Timeout / 5xx — back off and retry elsewhere. `scope` "model" means one
 * overloaded model (bench it, keep the key). */
export class Transient extends ProviderError {
  constructor(provider: string, status: number, detail = "", retryAfter: number | null = null, scope: ErrorScope = "key") {
    super(provider, status, detail, true, retryAfter, false, scope);
  }
}

/** This model can't serve the request on this provider — try another.
 * `gone` = the id is unknown/retired (404/410, "decommissioned", ...): the
 * provider benches it for a while. Otherwise only *this* request was rejected
 * for the model (e.g. context too long), so it stays in rotation. */
export class ModelNotFound extends ProviderError {
  /** `retired`: gone for everyone (410, "end of life", "decommissioned") vs.
   * unavailable to this key ("does not exist or you do not have access"). */
  constructor(provider: string, status: number, detail = "", public gone = false, public retired = false) {
    super(provider, status, detail, false, null, true);
  }
}

/** 402 — account out of credits/quota. Key gets disabled, call fails over. */
export class QuotaExhausted extends ProviderError {
  constructor(provider: string, status: number, detail = "") {
    super(provider, status, detail, false);
  }
}

/** This provider rejected the request itself (an unrecognised 4xx — a
 * parameter it doesn't support, content its moderation flagged, ...). Free
 * tiers disagree about what they accept, so the call fails over to the next
 * provider; once two providers reject the same request it's raised. */
export class BadRequest extends ProviderError {
  constructor(provider: string, status: number, detail = "") {
    super(provider, status, detail, false);
  }
}

/** Every candidate provider/key was exhausted or unavailable. `attempts` is
 * [[candidate, error], ...] for this call; `status` has one line per provider
 * explaining why it's unusable (disabled key, cooling down, ...). */
export class NoProvidersAvailable extends FreeLLMError {
  status: string[];
  constructor(public attempts: Array<[any, Error]>, status: string[] = []) {
    const detail = attempts
      .slice(0, 8)
      .map(([c, e]) => {
        const st = e instanceof ProviderError && e.status ? `(${e.status})` : "";
        return `${c.provider.name}/${c.model}:${e.constructor.name}${st}`;
      })
      .join("; ");
    let msg = `all providers/keys exhausted after ${attempts.length} attempt(s): ${detail || "none ready"}`;
    if (status.length) msg += `. Provider status: ${status.join(" | ")}`;
    if (attempts.some(([, e]) => e instanceof AuthError) || status.some((s) => s.includes("invalid/expired"))) {
      msg += ". Run `freelm doctor` to check your keys.";
    }
    super(msg);
    this.status = [...status];
  }
}

export function parseRetryAfter(value: string | null | undefined): number | null {
  if (!value) return null;
  const v = value.trim();
  if (/^\d+(\.\d+)?$/.test(v)) return Math.max(0, Number(v)); // decimal seconds only (no "0x10")
  const t = Date.parse(value);
  if (!Number.isNaN(t)) return Math.max(0, (t - Date.now()) / 1000);
  return null;
}

// Retryable statuses besides the whole 5xx range (Cloudflare 52x, 501, ...).
const TRANSIENT_STATUS = new Set([408, 409, 425]);
// 404 Not Found / 410 Gone: the model id is unknown or retired on this provider
// (e.g. NVIDIA NIM answers 410 for end-of-life models).
const GONE_STATUS = new Set([404, 410]);
// A model retired for everyone (vs. one this key can't access).
const RETIRED_HINTS = ["no longer", "decommission", "deprecated", "end of life", "end-of-life", "retired"];
// Phrases that mark a 400/422 "model" error as permanent for that model id.
const GONE_HINTS = [
  "no longer", "decommission", "deprecated", "end of life", "end-of-life", "retired",
  "not found", "not_found", "does not exist", "unknown model", "invalid model",
  "model not available", "no such model",
];
// The model exists but lacks a capability this request needs (tools, images, ...).
const CAPABILITY_HINTS = ["support", "tool use", "not enabled"];
// The request is too big for this model (another model/provider may take it).
const TOO_BIG_HINTS = [
  "context", "too long", "too large", "reduce the length", "maximum length",
  "token limit", "tokens per minute",
];
// Bad-key errors some providers send as 400 (Google: API_KEY_INVALID).
const AUTH_HINTS = [
  "api key not valid", "api_key_invalid", "api key expired", "api_key_expired",
  "invalid api key", "invalid_api_key", "incorrect api key",
];
// A 403 about the *content* (OpenRouter moderation), not the key.
const MODERATION_HINTS = ["flagged", "moderation"];
// A model this account's plan doesn't include (Cloudflare's Workers Paid-only
// models answer 403 on the Free plan) — the key itself is fine.
const PLAN_HINTS = ["workers paid", "paid plan"];
// A 429 meaning the quota is spent for the month (Cohere trial keys) — no
// recovery soon, like a 402 — or for the day (Cloudflare's daily free Neurons,
// OpenRouter's free-models-per-day): when no retry time is given, look again in
// an hour rather than every minute (daily resets aren't reliably at 00:00 UTC).
const MONTHLY_HINTS = ["/ month", "per month", "monthly limit", "monthly quota"];
const DAILY_HINTS = ["per day", "per-day", "daily free allocation", "daily limit", "daily quota"];
export const DAILY_QUOTA_RETRY = 3600;
// Google AI Studio refuses whole regions with a 400.
const LOCATION_HINTS = ["location is not supported", "unsupported_country", "not available in your country"];
// Google puts the server-suggested wait in the JSON body: "retryDelay": "36s"
const RETRY_DELAY_RE = /"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"/;

const has = (low: string, hints: string[]) => hints.some((h) => low.includes(h));

/** Map an HTTP error response to the right error. Nothing here aborts a call on
 * its own: even an unrecognised 4xx is a BadRequest that fails over (the engine
 * raises only once several providers reject the same request). */
export function classify(status: number, headers: Record<string, string> | null | undefined, body: string, provider: string): ProviderError {
  let retryAfter = parseRetryAfter(headers?.["retry-after"]);
  if (retryAfter === null && body) {
    const m = RETRY_DELAY_RE.exec(body);
    if (m) retryAfter = parseFloat(m[1]);
  }
  const msg = (body || "").slice(0, 300);
  const low = (body || "").toLowerCase();
  if ((status === 400 || status === 401 || status === 403) && has(low, AUTH_HINTS)) return new AuthError(provider, status, msg);
  if (status === 403 && has(low, MODERATION_HINTS)) return new BadRequest(provider, status, msg);
  if (status === 403 && has(low, PLAN_HINTS)) return new ModelNotFound(provider, status, msg, true); // not on this account's plan
  if (status === 401 || status === 403) return new AuthError(provider, status, msg);
  if (status === 402) return new QuotaExhausted(provider, status, msg);
  if (status === 429) {
    if (has(low, MONTHLY_HINTS)) return new QuotaExhausted(provider, status, msg);
    if (retryAfter === null && has(low, DAILY_HINTS)) retryAfter = DAILY_QUOTA_RETRY;
    return new RateLimited(provider, status, msg, retryAfter);
  }
  if (TRANSIENT_STATUS.has(status) || (status >= 500 && status <= 599)) return new Transient(provider, status, msg, retryAfter);
  if (GONE_STATUS.has(status)) {
    // OpenRouter 404s "No endpoints found that support tool use": the model is
    // fine, it just can't do what this request asks — don't bench it
    const gone = !has(low, CAPABILITY_HINTS);
    return new ModelNotFound(provider, status, msg, gone, gone && (status === 410 || has(low, RETIRED_HINTS)));
  }
  if (status === 400 || status === 413 || status === 422) {
    if (low.includes("model") && has(low, GONE_HINTS)) return new ModelNotFound(provider, status, msg, true, has(low, RETIRED_HINTS));
    if (status === 413 || has(low, TOO_BIG_HINTS) || low.includes("model")) {
      return new ModelNotFound(provider, status, msg); // too big / unsupported for this model only
    }
  }
  if (has(low, LOCATION_HINTS)) return new AuthError(provider, status, msg); // key unusable from here
  return new BadRequest(provider, status, msg);
}
