"""Pure (no-I/O) orchestration helpers shared by the sync and async clients.

The actual HTTP call differs between sync/async, but candidate selection and
post-attempt state updates are identical, so they live here.
"""
from __future__ import annotations

from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple, Union

from ._backoff import compute_delay
from .errors import (
    AuthError,
    BadRequest,
    ModelNotFound,
    NoProvidersAvailable,
    ProviderError,
    QuotaExhausted,
    RateLimited,
    Transient,
)
from .strategy import Candidate, order_candidates, routable

TriedKey = Tuple[str, str, str]

# How long a retired model (404/410 "gone") stays benched before we try it again.
MODEL_GONE_TTL = 3600.0
# Default bench for a model throttled upstream when no Retry-After is given.
MODEL_THROTTLE_TTL = 60.0
# Bench for a model reported overloaded (model-scoped 5xx).
MODEL_OVERLOAD_TTL = 30.0
# A request rejected (BadRequest) by this many distinct providers is a caller bug.
REJECTIONS_TO_RAISE = 2
# Hedging: an attempt still running after this long gets a parallel one on the
# next candidate, and the first answer wins. Adaptive: 3x the key's latency
# average, clamped — (floor, cap, unmeasured) in seconds. Streams race to the
# first token, so they hedge sooner than whole responses.
HEDGE_STREAM = (1.5, 6.0, 3.0)
HEDGE_CHAT = (4.0, 12.0, 6.0)


def hedge_delay(cand: Candidate, setting: Union[bool, float, None], stream: bool) -> Optional[float]:
    """Seconds after which a still-running attempt on ``cand`` gets a parallel
    hedge, or None. ``setting`` is the client's ``hedge``: True = adaptive, a
    number = fixed seconds, False/None/0 = off."""
    if setting is None or setting is False:
        return None
    if setting is not True:
        return float(setting) if setting > 0 else None
    floor, cap, unmeasured = HEDGE_STREAM if stream else HEDGE_CHAT
    ewma = cand.key.ewma_latency
    if ewma <= 0:
        return unmeasured
    return min(cap, max(floor, 3.0 * ewma / 1000.0))


def select_candidate(
    providers: List[Any],
    strategy: str,
    rr: Dict[str, int],
    alias: Union[str, Sequence[str]],
    tried: Set[TriedKey],
    now: float,
    skip: Collection[str] = (),
) -> Optional[Candidate]:
    """First ready candidate not already tried this call, in strategy order.
    ``skip`` holds provider names that already rejected this request."""
    for c in order_candidates(providers, alias, now, strategy, rr):
        if c.provider.name in skip:
            continue
        if (c.provider.name, c.key.key, c.model) in tried:
            continue
        if not c.provider.model_ready(c.model, now) or not c.key.model_ready(c.model, now):
            continue  # benched model (for everyone, or just for this key)
        if c.key.ready(now):
            return c
    return None


def forget_recovered(providers: List[Any], tried: Set[TriedKey], now: float) -> Set[TriedKey]:
    """Drop ``tried`` entries whose key is ready again, so a key that was cooling
    can be retried after a ``wait``. Bounded by ``max_attempts`` in the caller."""
    ready_keys = {(p.name, k.key) for p in providers for k in p.keys if k.ready(now)}
    return {t for t in tried if (t[0], t[1]) not in ready_keys}


def soonest_wait(
    providers: List[Any],
    now: float,
    alias: Optional[Union[str, Sequence[str]]] = None,
    tried: Collection[TriedKey] = (),
    skip: Collection[str] = (),
) -> Optional[float]:
    """Seconds until some candidate for ``alias`` could be tried again: the
    largest of its key's wait and its model benches (provider-wide and
    per-key). Candidates already tried whose state can't change are ignored, so
    0 never comes back — None means waiting can't help. Without ``alias`` it is
    the soonest any non-disabled key frees up."""
    if alias is None:
        key_waits = [w for p in providers for k in p.keys for w in (k.wait_time(now),) if w is not None]
        return min(key_waits) if key_waits else None
    waits: List[float] = []
    routes, _ = routable([p for p in providers if p.name not in skip], alias)
    for p, models in routes:
        for k in p.keys:
            kw = k.wait_time(now)
            if kw is None:
                continue  # disabled
            for m in models:
                w = max(kw, p.model_wait(m, now), k.model_wait(m, now))
                if w > 0:
                    waits.append(w)
    return min(waits) if waits else None


def provider_status(providers: List[Any], now: float) -> List[str]:
    """One line per provider that can't serve right now, saying why — appended
    to ``NoProvidersAvailable`` so a failed call is self-explaining."""
    out: List[str] = []
    for p in providers:
        keys = list(p.keys)
        if keys and all(k.disabled for k in keys):
            errs = sorted({k.last_error or "disabled" for k in keys})
            why = ", ".join(errs)
            if any(e.startswith("auth") for e in errs):
                why += " — key invalid/expired?"
            elif any(e.startswith("quota") for e in errs):
                why += " — out of credits/quota"
            out.append(f"{p.name}: disabled ({why})")
        elif not any(k.ready(now) for k in keys):
            waits = [w for w in (k.wait_time(now) for k in keys) if w is not None]
            errs = sorted({k.last_error for k in keys if k.last_error})
            soon = f" ~{max(1, round(min(waits)))}s" if waits else ""
            out.append(f"{p.name}: cooling down{soon}" + (f" ({', '.join(errs)})" if errs else ""))
        elif p.models and not any(
            k.ready(now) and p.model_ready(m.id, now) and k.model_ready(m.id, now) for k in keys for m in p.models
        ):
            out.append(f"{p.name}: no usable model (all benched)")
    return out


def apply_success(cand: Candidate, latency_ms: float, now: Optional[float] = None) -> None:
    k = cand.key
    k.breaker.on_success()
    k.last_error = None
    if latency_ms > 0:  # <=0 means "no sample" (e.g. an empty stream) — don't decay the EWMA
        k.ewma_latency = latency_ms if k.ewma_latency == 0 else 0.7 * k.ewma_latency + 0.3 * latency_ms
        if now is not None:
            k.latency_at = now


def apply_slow(cand: Candidate, elapsed_ms: float, now: float) -> None:
    """A hedge beat this attempt. Not an error — but the key was at least this
    slow, so ``smart`` routing puts it behind faster ones for a while."""
    k = cand.key
    k.ewma_latency = max(k.ewma_latency, elapsed_ms)
    k.latency_at = now


def _refund(k: Any) -> None:
    if k.rpd is not None and k.rpd_used > 0:
        k.rpd_used -= 1  # the request never reached inference -> give the daily slot back


def apply_error(cand: Candidate, exc: ProviderError, now: float) -> None:
    """Update key/model state after a failed attempt. Returns nothing; raising is
    the caller's decision (see ``should_raise``)."""
    k = cand.key
    if isinstance(exc, AuthError):
        k.disabled = True
        k.last_error = f"auth:{exc.status}"
    elif isinstance(exc, QuotaExhausted):
        if cand.model and cand.model.lower() in (exc.message or "").lower():
            # "payment required for <model>": that model is paid for this account
            k.bench_model(cand.model, now + MODEL_GONE_TTL)
            k.last_error = f"quota:{exc.status}:model"
        else:
            k.disabled = True  # out of credits — won't recover without human action
            k.last_error = f"quota:{exc.status}"
    elif isinstance(exc, RateLimited):
        wait = exc.retry_after  # NB: 0 is a valid answer ("retry now"), so no `or default`
        until = now + (wait if wait is not None else MODEL_THROTTLE_TTL)
        scope = getattr(exc, "scope", "key")
        if scope == "upstream":
            # the model is throttled for everyone (OpenRouter's shared free
            # pool): bench it on every key, keep the keys hot
            cand.provider.bench_model(cand.model, until)
            k.last_error = "rate_limited:upstream"
        elif scope == "model":
            # this key's quota for this model (Gemini, Groq): other models on
            # the key and this model on other keys stay usable
            k.bench_model(cand.model, until)
            k.last_error = "rate_limited:model"
        else:
            k.cooldown_until = now + (wait if wait is not None else 60.0)
            k.last_error = "rate_limited"
    elif isinstance(exc, ModelNotFound):
        k.last_error = "model_missing"  # don't penalise the key for a bad model id
        if getattr(exc, "gone", False):
            # stop offering it instead of re-trying it first on every call:
            # for everyone when retired, else just for this key (no access)
            if getattr(exc, "retired", False):
                cand.provider.bench_model(cand.model, now + MODEL_GONE_TTL)
            else:
                k.bench_model(cand.model, now + MODEL_GONE_TTL)
        _refund(k)
    elif isinstance(exc, Transient):
        if getattr(exc, "scope", "key") == "model":
            # one model overloaded upstream (e.g. Gemini "high demand") — the
            # key is fine, so bench only the model and move on
            ttl = exc.retry_after if exc.retry_after is not None else MODEL_OVERLOAD_TTL
            cand.provider.bench_model(cand.model, now + min(MODEL_OVERLOAD_TTL, ttl))
            k.last_error = f"transient:{exc.status}:model"
            return
        k.breaker.on_failure(now)
        delay = exc.retry_after if exc.retry_after is not None else compute_delay(k.breaker.failures)
        k.cooldown_until = now + min(30.0, delay)
        k.last_error = f"transient:{exc.status}"
    elif isinstance(exc, BadRequest):
        # the provider refused this request; nothing is wrong with the key
        k.last_error = f"rejected:{exc.status}"
        _refund(k)
    else:  # an unclassified ProviderError (custom providers)
        k.breaker.on_failure(now)
        k.last_error = f"error:{exc.status}"


def rejected_by(attempts: Sequence[Tuple[Any, Exception]]) -> Set[str]:
    """Names of the providers that rejected this request (``BadRequest``)."""
    return {c.provider.name for c, e in attempts if isinstance(e, BadRequest)}


def should_raise(exc: ProviderError, attempts: Sequence[Tuple[Any, Exception]] = ()) -> bool:
    """Abort the call now instead of failing over?

    Only a ``BadRequest`` can abort, and only once ``REJECTIONS_TO_RAISE``
    different providers rejected the same request — then it's the caller's bug,
    not one provider's quirk. Everything else (auth, quota, rate limits,
    retired models, 5xx) fails over."""
    if isinstance(exc, BadRequest):
        return len(rejected_by(attempts)) >= REJECTIONS_TO_RAISE
    if isinstance(exc, (AuthError, QuotaExhausted, RateLimited, Transient, ModelNotFound)):
        return False
    return not exc.retryable and not exc.model_missing


def exhausted(
    attempts: Sequence[Tuple[Any, Exception]], providers: List[Any], now: float
) -> Exception:
    """The error to raise when the loop runs out of candidates: the providers'
    own rejection when every attempt was one *and* every provider rejected it
    (the request itself is the problem — e.g. a one-provider setup), else
    ``NoProvidersAvailable`` with a per-provider explanation (others may just
    have been cooling down)."""
    if (
        attempts
        and all(isinstance(e, BadRequest) for _, e in attempts)
        and rejected_by(attempts) >= {p.name for p in providers}
    ):
        return attempts[0][1]
    return NoProvidersAvailable(list(attempts), provider_status(providers, now))
