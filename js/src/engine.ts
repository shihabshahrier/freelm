/** Pure orchestration helpers shared by the client's request loops. */
import { computeDelay } from "./backoff.js";
import {
  AuthError,
  BadRequest,
  ModelNotFound,
  NoProvidersAvailable,
  ProviderError,
  QuotaExhausted,
  RateLimited,
  Transient,
} from "./errors.js";
import { Candidate, orderCandidates, routable } from "./strategy.js";

/** How long a retired model (404/410 "gone") stays benched before we try it again. */
export const MODEL_GONE_TTL = 3600;
/** Default bench for a model throttled upstream when no Retry-After is given. */
export const MODEL_THROTTLE_TTL = 60;
/** Bench for a model reported overloaded (model-scoped 5xx). */
export const MODEL_OVERLOAD_TTL = 30;
/** A request rejected (BadRequest) by this many distinct providers is a caller bug. */
export const REJECTIONS_TO_RAISE = 2;
/** Hedging: an attempt still running after this long gets a parallel one on the
 * next candidate, and the first answer wins. Adaptive: 3x the key's latency
 * average, clamped — [floor, cap, unmeasured] in seconds. Streams race to the
 * first token, so they hedge sooner than whole responses. */
export const HEDGE_STREAM: readonly [number, number, number] = [1.5, 6, 3];
export const HEDGE_CHAT: readonly [number, number, number] = [4, 12, 6];

/** Seconds after which a still-running attempt on `c` gets a parallel hedge,
 * or null. `setting` is the client's `hedge`: true = adaptive, a number = fixed
 * seconds, false/0 = off. */
export function hedgeDelay(c: Candidate, setting: boolean | number | null | undefined, stream: boolean): number | null {
  if (setting === null || setting === undefined || setting === false) return null;
  if (setting !== true) return setting > 0 ? setting : null;
  const [floor, cap, unmeasured] = stream ? HEDGE_STREAM : HEDGE_CHAT;
  const ewma = c.key.ewmaLatency;
  if (ewma <= 0) return unmeasured;
  return Math.min(cap, Math.max(floor, (3 * ewma) / 1000));
}

const SEP = "\u0000";

export function triedKey(c: Candidate): string {
  return `${c.provider.name}${SEP}${c.key.key}${SEP}${c.model}`;
}

/** First ready candidate not already tried this call, in strategy order.
 * `skip` holds provider names that already rejected this request. */
export function selectCandidate(
  providers: any[],
  strategy: string,
  rr: { p: number },
  alias: string | string[],
  tried: Set<string>,
  now: number,
  skip: Set<string> = new Set(),
): Candidate | null {
  for (const c of orderCandidates(providers, alias, now, strategy, rr)) {
    if (skip.has(c.provider.name)) continue;
    if (tried.has(triedKey(c))) continue;
    if (!c.provider.modelReady(c.model, now) || !c.key.modelReady(c.model, now)) continue; // benched (for all, or this key)
    if (c.key.ready(now)) return c;
  }
  return null;
}

/** Drop tried entries whose key is ready again (so a cooled key can be retried after a wait). */
export function forgetRecovered(providers: any[], tried: Set<string>, now: number): Set<string> {
  const readyKeys = new Set<string>();
  for (const p of providers) for (const k of p.keys) if (k.ready(now)) readyKeys.add(`${p.name}${SEP}${k.key}`);
  const out = new Set<string>();
  for (const t of tried) {
    const parts = t.split(SEP);
    if (!readyKeys.has(`${parts[0]}${SEP}${parts[1]}`)) out.add(t);
  }
  return out;
}

/** Seconds until some candidate for `alias` could be tried again: the largest
 * of its key's wait and its model benches (provider-wide and per-key). Never 0
 * — null means waiting can't help. Without `alias`: the soonest any
 * non-disabled key frees up. */
export function soonestWait(
  providers: any[],
  now: number,
  alias?: string | string[],
  _tried: Set<string> = new Set(),
  skip: Set<string> = new Set(),
): number | null {
  const waits: number[] = [];
  if (alias === undefined) {
    for (const p of providers)
      for (const k of p.keys) {
        const w = k.waitTime(now);
        if (w !== null) waits.push(w);
      }
    return waits.length ? Math.min(...waits) : null;
  }
  const { routes } = routable(providers.filter((p) => !skip.has(p.name)), alias);
  for (const [p, models] of routes)
    for (const k of p.keys) {
      const kw = k.waitTime(now);
      if (kw === null) continue; // disabled
      for (const m of models) {
        const w = Math.max(kw, p.modelWait(m, now), k.modelWait(m, now));
        if (w > 0) waits.push(w);
      }
    }
  return waits.length ? Math.min(...waits) : null;
}

/** One line per provider that can't serve right now, saying why — appended to
 * NoProvidersAvailable so a failed call is self-explaining. */
export function providerStatus(providers: any[], now: number): string[] {
  const out: string[] = [];
  for (const p of providers) {
    const keys: any[] = [...p.keys];
    if (keys.length && keys.every((k) => k.disabled)) {
      const errs = [...new Set(keys.map((k) => k.lastError ?? "disabled"))].sort();
      let why = errs.join(", ");
      if (errs.some((e) => e.startsWith("auth"))) why += " — key invalid/expired?";
      else if (errs.some((e) => e.startsWith("quota"))) why += " — out of credits/quota";
      out.push(`${p.name}: disabled (${why})`);
    } else if (!keys.some((k) => k.ready(now))) {
      const waits = keys.map((k) => k.waitTime(now)).filter((w): w is number => w !== null);
      const errs = [...new Set(keys.map((k) => k.lastError).filter(Boolean))].sort();
      const soon = waits.length ? ` ~${Math.max(1, Math.round(Math.min(...waits)))}s` : "";
      out.push(`${p.name}: cooling down${soon}${errs.length ? ` (${errs.join(", ")})` : ""}`);
    } else if (p.models.length && !keys.some((k) => k.ready(now) && p.models.some((m: any) => p.modelReady(m.id, now) && k.modelReady(m.id, now)))) {
      out.push(`${p.name}: no usable model (all benched)`);
    }
  }
  return out;
}

export function applySuccess(c: Candidate, latencyMs: number, now?: number): void {
  const k = c.key;
  k.breaker.onSuccess();
  k.lastError = null;
  if (latencyMs > 0) {
    // <=0 means "no sample" (e.g. an empty stream) — don't decay the EWMA
    k.ewmaLatency = k.ewmaLatency === 0 ? latencyMs : 0.7 * k.ewmaLatency + 0.3 * latencyMs;
    if (now !== undefined) k.latencyAt = now;
  }
}

/** A hedge beat this attempt. Not an error — but the key was at least this
 * slow, so `smart` routing puts it behind faster ones for a while. */
export function applySlow(c: Candidate, elapsedMs: number, now: number): void {
  c.key.ewmaLatency = Math.max(c.key.ewmaLatency, elapsedMs);
  c.key.latencyAt = now;
}

function refund(k: any): void {
  if (k.rpd !== null && k.rpdUsed > 0) k.rpdUsed--; // never reached inference -> give the daily slot back
}

/** Update key/model state after a failed attempt; raising is the caller's call. */
export function applyError(c: Candidate, exc: ProviderError, now: number): void {
  const k = c.key;
  if (exc instanceof AuthError) {
    k.disabled = true;
    k.lastError = `auth:${exc.status}`;
  } else if (exc instanceof QuotaExhausted) {
    if (c.model && (exc.detail || "").toLowerCase().includes(c.model.toLowerCase())) {
      // "payment required for <model>": that model is paid for this account
      k.benchModel(c.model, now + MODEL_GONE_TTL);
      k.lastError = `quota:${exc.status}:model`;
    } else {
      k.disabled = true; // out of credits — won't recover without human action
      k.lastError = `quota:${exc.status}`;
    }
  } else if (exc instanceof RateLimited) {
    const until = now + (exc.retryAfter ?? MODEL_THROTTLE_TTL);
    if (exc.scope === "upstream") {
      // the model is throttled for everyone (OpenRouter's shared free pool)
      c.provider.benchModel(c.model, until);
      k.lastError = "rate_limited:upstream";
    } else if (exc.scope === "model") {
      // this key's quota for this model (Gemini, Groq): other models on the
      // key and this model on other keys stay usable
      k.benchModel(c.model, until);
      k.lastError = "rate_limited:model";
    } else {
      k.cooldownUntil = now + (exc.retryAfter ?? 60);
      k.lastError = "rate_limited";
    }
  } else if (exc instanceof ModelNotFound) {
    k.lastError = "model_missing"; // don't penalise the key for a bad model id
    // stop offering it instead of re-trying it first on every call: for
    // everyone when retired, else just for this key (no access)
    if (exc.gone) {
      if (exc.retired) c.provider.benchModel(c.model, now + MODEL_GONE_TTL);
      else k.benchModel(c.model, now + MODEL_GONE_TTL);
    }
    refund(k);
  } else if (exc instanceof Transient) {
    if (exc.scope === "model") {
      // one model overloaded upstream (e.g. Gemini "high demand") — bench only it
      c.provider.benchModel(c.model, now + Math.min(MODEL_OVERLOAD_TTL, exc.retryAfter ?? MODEL_OVERLOAD_TTL));
      k.lastError = `transient:${exc.status}:model`;
      return;
    }
    k.breaker.onFailure(now);
    const delay = exc.retryAfter ?? computeDelay(k.breaker.failures);
    k.cooldownUntil = now + Math.min(30, delay);
    k.lastError = `transient:${exc.status}`;
  } else if (exc instanceof BadRequest) {
    k.lastError = `rejected:${exc.status}`; // the provider refused this request; the key is fine
    refund(k);
  } else {
    k.breaker.onFailure(now); // an unclassified ProviderError (custom providers)
    k.lastError = `error:${exc.status}`;
  }
}

/** Names of the providers that rejected this request (BadRequest). */
export function rejectedBy(attempts: Array<[Candidate, Error]>): Set<string> {
  return new Set(attempts.filter(([, e]) => e instanceof BadRequest).map(([c]) => c.provider.name));
}

/** Abort the call now instead of failing over? Only a BadRequest can, and only
 * once REJECTIONS_TO_RAISE different providers rejected the same request —
 * then it's the caller's bug, not one provider's quirk. */
export function shouldRaise(exc: ProviderError, attempts: Array<[Candidate, Error]> = []): boolean {
  if (exc instanceof BadRequest) return rejectedBy(attempts).size >= REJECTIONS_TO_RAISE;
  if (
    exc instanceof AuthError ||
    exc instanceof QuotaExhausted ||
    exc instanceof RateLimited ||
    exc instanceof Transient ||
    exc instanceof ModelNotFound
  ) {
    return false;
  }
  return !exc.retryable && !exc.modelMissing;
}

/** The error to throw when the loop runs out of candidates: the providers' own
 * rejection if every attempt was one, else NoProvidersAvailable with a
 * per-provider explanation. */
export function exhausted(attempts: Array<[Candidate, Error]>, providers: any[], now: number): Error {
  // the providers' own rejection only when every provider rejected it (e.g. a
  // one-provider setup); otherwise others may just have been cooling down
  const rejected = rejectedBy(attempts);
  if (attempts.length && attempts.every(([, e]) => e instanceof BadRequest) && providers.every((p) => rejected.has(p.name))) {
    return attempts[0][1];
  }
  return new NoProvidersAvailable(attempts, providerStatus(providers, now));
}
