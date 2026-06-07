"""Candidate ordering strategies.

A *candidate* is one concrete (provider, key, model) we could try. Strategies
decide the order; the engine walks them and picks the first that is ready.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Dict, List

PRIORITY = "priority"
ROUND_ROBIN = "round_robin"
QUOTA_AWARE = "quota_aware"
LATENCY = "latency"
STRATEGIES = (PRIORITY, ROUND_ROBIN, QUOTA_AWARE, LATENCY)


@dataclass
class Candidate:
    provider: Any  # providers.base.Provider
    key: Any       # _keys.KeyState
    model: str


def order_candidates(
    providers: List[Any],
    alias: str,
    now: float,
    strategy: str,
    rr: Dict[str, int],
) -> List[Candidate]:
    """Ordered list of (provider, key, model) candidates to try.

    Providers are ranked by ``strategy``; candidates are then **interleaved
    breadth-first across providers** — i.e. the best model of every provider is
    tried before any provider's second model. This guarantees failover reaches
    every provider quickly instead of burning all attempts on one provider's
    many (possibly throttled) models.
    """
    provs = list(providers)

    if strategy == ROUND_ROBIN and provs:
        i = rr.get("p", 0) % len(provs)
        provs = provs[i:] + provs[:i]
        rr["p"] = rr.get("p", 0) + 1
    elif strategy == QUOTA_AWARE:
        provs.sort(key=lambda p: p.capacity(now), reverse=True)
    elif strategy == LATENCY:
        provs.sort(key=lambda p: p.avg_latency())
    else:  # PRIORITY
        provs.sort(key=lambda p: p.priority)

    # Build each provider's own ordered sublist (rotated keys, then models).
    per_provider: List[List[Candidate]] = []
    for p in provs:
        keys = list(p.keys)
        if keys:  # rotate keys within a provider to spread load
            ki = p._rr % len(keys)
            keys = keys[ki:] + keys[:ki]
            p._rr += 1
        models = p.resolve_models(alias)
        sub = [Candidate(p, k, mid) for k in keys for mid in models]
        if sub:
            per_provider.append(sub)

    # Interleave: rank-0 of every provider first (in strategy order), then rank-1, ...
    candidates: List[Candidate] = []
    for rank in zip_longest(*per_provider):
        for c in rank:
            if c is not None:
                candidates.append(c)
    return candidates
