"""Pure (no-I/O) orchestration helpers shared by the sync and async clients.

The actual HTTP call differs between sync/async, but candidate selection and
post-attempt state updates are identical, so they live here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from ._backoff import compute_delay
from .errors import AuthError, ModelNotFound, ProviderError, QuotaExhausted, RateLimited, Transient
from .strategy import Candidate, order_candidates

TriedKey = Tuple[str, str, str]


def select_candidate(
    providers: List[Any],
    strategy: str,
    rr: Dict[str, int],
    alias: str,
    tried: Set[TriedKey],
    now: float,
) -> Optional[Candidate]:
    """First ready candidate not already tried this call, in strategy order."""
    for c in order_candidates(providers, alias, now, strategy, rr):
        if (c.provider.name, c.key.key, c.model) in tried:
            continue
        if c.key.ready(now):
            return c
    return None


def forget_recovered(providers: List[Any], tried: Set[TriedKey], now: float) -> Set[TriedKey]:
    """Drop ``tried`` entries whose key is ready again, so a key that was cooling
    can be retried after a ``wait``. Bounded by ``max_attempts`` in the caller."""
    ready_keys = {(p.name, k.key) for p in providers for k in p.keys if k.ready(now)}
    return {t for t in tried if (t[0], t[1]) not in ready_keys}


def soonest_wait(providers: List[Any], now: float) -> Optional[float]:
    """Smallest wait until *some* non-disabled key becomes ready, or None."""
    waits: List[float] = []
    for p in providers:
        for k in p.keys:
            w = k.wait_time(now)
            if w is not None:
                waits.append(w)
    return min(waits) if waits else None


def apply_success(cand: Candidate, latency_ms: float) -> None:
    k = cand.key
    k.breaker.on_success()
    k.last_error = None
    if latency_ms > 0:  # <=0 means "no sample" (e.g. an empty stream) — don't decay the EWMA
        k.ewma_latency = latency_ms if k.ewma_latency == 0 else 0.7 * k.ewma_latency + 0.3 * latency_ms


def apply_error(cand: Candidate, exc: ProviderError, now: float) -> None:
    """Update key state after a failed attempt. Returns nothing; raising is the
    caller's decision (see ``should_raise``)."""
    k = cand.key
    if isinstance(exc, AuthError):
        k.disabled = True
        k.last_error = f"auth:{exc.status}"
    elif isinstance(exc, QuotaExhausted):
        k.disabled = True  # out of credits — won't recover without human action
        k.last_error = f"quota:{exc.status}"
    elif isinstance(exc, RateLimited):
        if getattr(exc, "scope", "key") == "model":
            # only this model is throttled upstream — keep the key hot, the
            # 'tried' set steers us to the next model on the same key.
            k.last_error = "rate_limited:model"
        else:
            k.cooldown_until = now + (exc.retry_after or 60.0)
            k.last_error = "rate_limited"
    elif isinstance(exc, ModelNotFound):
        k.last_error = "model_missing"  # don't penalise the key for a bad model id
        if k.rpd is not None and k.rpd_used > 0:
            k.rpd_used -= 1  # a 404 never hit inference -> refund the daily slot
    elif isinstance(exc, Transient):
        k.breaker.on_failure(now)
        delay = exc.retry_after if exc.retry_after is not None else compute_delay(k.breaker.failures)
        k.cooldown_until = now + min(30.0, delay)
        k.last_error = f"transient:{exc.status}"
    else:  # non-retryable ProviderError
        k.breaker.on_failure(now)
        k.last_error = f"error:{exc.status}"


def should_raise(exc: ProviderError) -> bool:
    """A non-retryable, non-model error (e.g. malformed 400/422) is a caller bug:
    bail immediately instead of burning every key on the same broken request.

    Auth/quota errors are *not* fatal to the whole call — the key is disabled
    and we fail over to other keys/providers."""
    if isinstance(exc, (AuthError, QuotaExhausted)):
        return False
    return not exc.retryable and not exc.model_missing
