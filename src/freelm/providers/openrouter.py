"""OpenRouter (https://openrouter.ai) — OpenAI-compatible.

Free models carry a ``:free`` suffix. Daily caps depend on credit balance:
~50 free requests/day under $10 lifetime credit, ~1000/day at >= $10. Pick the
matching tier or override ``rpd``.
"""
from __future__ import annotations

from typing import Any, Dict

from ..registry import ModelSpec
from .base import Provider


class OpenRouter(Provider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    TIERS: Dict[str, Dict[str, Any]] = {
        "free": {"rpm": 20, "rpd": 50},        # < $10 lifetime credit
        "credit": {"rpm": 20, "rpd": 1000},    # >= $10 lifetime credit
    }

    # Free model ids churn constantly; these were listed as free in the public
    # catalog on 2026-10-09 (diverse upstreams, so one throttle still fails over).
    # Live discovery replaces this list at runtime.
    DEFAULT_MODELS = [
        ModelSpec("google/gemma-4-31b-it:free", ("chat", "large", "tools", "vision"), ctx=262144),
        ModelSpec("nvidia/nemotron-3-super-120b-a12b:free", ("chat", "large", "tools", "reasoning"), ctx=262144),
        ModelSpec("google/gemma-4-26b-a4b-it:free", ("chat", "fast", "tools", "vision"), ctx=262144),
        ModelSpec("nvidia/nemotron-3.5-lightning:free", ("chat", "fast", "tools"), ctx=1000000),
        ModelSpec("poolside/laguna-s-2.1:free", ("chat", "tools"), ctx=262144),
        ModelSpec("thinkingmachines/inkling-small:free", ("chat", "small", "fast", "tools", "vision"), ctx=1048576),
        # OpenRouter's own router across whatever free models are up right now
        ModelSpec("openrouter/free", ("chat",), ctx=200000),
    ]

    def __init__(self, keys, **kw):
        # App attribution: OpenRouter lists apps that send a referer + title on
        # openrouter.ai/apps and in each model's "Apps" tab. Override via extra_headers.
        extra = {"HTTP-Referer": "https://github.com/shihabshahrier/freelm", "X-Title": "freelm"}
        extra.update(kw.pop("extra_headers", None) or {})
        # Free models churn constantly -> discover live by default, free-only.
        kw.setdefault("discover", not kw.get("models"))  # an explicit models= list wins
        kw.setdefault("discover_free_only", True)
        # OpenRouter's catalog mixes paid and free models -> guard paid ids.
        kw.setdefault("free_only", True)
        super().__init__(keys, extra_headers=extra, **kw)

    def rate_limit_scope(self, body: str) -> str:
        b = (body or "").lower()
        # e.g. "<model> is temporarily rate-limited upstream". Deliberately narrow:
        # a bare "temporarily" also appears in account-wide 429s, which must cool
        # the key instead of hammering it with the next model.
        if "rate-limited upstream" in b:
            return "upstream"  # the free model is throttled for everyone, not just this key
        return "key"
