"""Candidate ordering strategies.

A *candidate* is one concrete (provider, key, model) we could try. Strategies
decide the order; the engine walks them and picks the first that is ready.
"""
from __future__ import annotations

from dataclasses import dataclass
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

    candidates: List[Candidate] = []
    for p in provs:
        keys = list(p.keys)
        if keys:  # rotate keys within a provider to spread load
            ki = p._rr % len(keys)
            keys = keys[ki:] + keys[:ki]
            p._rr += 1
        models = p.resolve_models(alias)
        for k in keys:
            for mid in models:
                candidates.append(Candidate(p, k, mid))
    return candidates
