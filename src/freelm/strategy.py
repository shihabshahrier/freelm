"""Candidate ordering strategies.

A *candidate* is one concrete (provider, key, model) we could try. Strategies
decide the order; the engine walks them and picks the first that is ready.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .errors import ConfigError
from .registry import is_virtual

SMART = "smart"
PRIORITY = "priority"
ROUND_ROBIN = "round_robin"
QUOTA_AWARE = "quota_aware"
LATENCY = "latency"
STRATEGIES = (SMART, PRIORITY, ROUND_ROBIN, QUOTA_AWARE, LATENCY)


@dataclass
class Candidate:
    provider: Any  # providers.base.Provider
    key: Any       # _keys.KeyState
    model: str


def routable(
    providers: Sequence[Any], alias: Union[str, Sequence[str]]
) -> Tuple[List[Tuple[Any, List[str]]], Optional[ConfigError]]:
    """``[(provider, model_ids), ...]`` this alias may use, in the given order.

    A concrete model id goes only to the providers that list it; an id no
    provider lists is passed through to all of them (it may still exist). A
    provider whose free guard rejects the id is skipped; its ``ConfigError``
    is returned so the caller can raise it when nothing else can serve."""
    provs = list(providers)
    aliases = [alias] if isinstance(alias, str) else list(alias)
    owners = {a: {id(p) for p in provs if p.knows_model(a)} for a in aliases if not is_virtual(a)}
    out: List[Tuple[Any, List[str]]] = []
    guard: Optional[ConfigError] = None
    for p in provs:
        chain = [a for a in aliases if not owners.get(a) or id(p) in owners[a]]
        if not chain:
            continue
        try:
            models = p.resolve_models(chain)
        except ConfigError as e:
            guard = guard or e
            continue
        if models:
            out.append((p, models))
    return out, guard


def order_candidates(
    providers: List[Any],
    alias: Union[str, Sequence[str]],
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

    # provider ``priority`` is the universal tiebreak: primary for PRIORITY,
    # secondary for the dynamic strategies, baseline order for ROUND_ROBIN.
    if strategy == ROUND_ROBIN and provs:
        provs.sort(key=lambda p: p.priority)
        i = rr.get("p", 0) % len(provs)
        provs = provs[i:] + provs[:i]
        rr["p"] = rr.get("p", 0) + 1
    elif strategy == QUOTA_AWARE:
        provs.sort(key=lambda p: (-p.capacity(now), p.priority))
    elif strategy == LATENCY:
        provs.sort(key=lambda p: (p.avg_latency(), p.priority))
    elif strategy == SMART:
        # priority tiers first; within a tier the fastest measured provider —
        # an unknown one counts as typical, so it gets tried and measured
        provs.sort(key=lambda p: (p.priority, p.expected_latency(now)))
    else:  # PRIORITY
        provs.sort(key=lambda p: p.priority)

    routes, guard = routable(provs, alias)

    # Build each provider's own ordered sublist (rotated keys, then models).
    # NB: benched models keep their rank slot here and are skipped at selection
    # time — dropping them would promote the provider's next (often equally
    # dead) model to rank 0 and starve the interleave.
    per_provider: List[List[Candidate]] = []
    for p, models in routes:
        keys = list(p.keys)
        if keys:  # rotate keys within a provider to spread load
            ki = p._rr % len(keys)
            keys = keys[ki:] + keys[:ki]
            p._rr += 1
        sub = [Candidate(p, k, mid) for k in keys for mid in models]
        if sub:
            per_provider.append(sub)
    if not per_provider and guard is not None:
        raise guard

    # Interleave: rank-0 of every provider first (in strategy order), then rank-1, ...
    candidates: List[Candidate] = []
    for rank in zip_longest(*per_provider):
        for c in rank:
            if c is not None:
                candidates.append(c)
    return candidates
