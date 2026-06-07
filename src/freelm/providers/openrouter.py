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

    # Free model ids churn constantly; these were verified live 2026-06-07.
    # Diverse upstreams are listed so a single upstream throttle still fails over.
    DEFAULT_MODELS = [
        ModelSpec("openai/gpt-oss-120b:free", ("chat", "large"), ctx=131072),
        ModelSpec("openai/gpt-oss-20b:free", ("chat", "small", "fast"), ctx=131072),
        ModelSpec("meta-llama/llama-3.3-70b-instruct:free", ("chat", "large"), ctx=131072),
        ModelSpec("z-ai/glm-4.5-air:free", ("chat", "large"), ctx=131072),
        ModelSpec("qwen/qwen3-next-80b-a3b-instruct:free", ("chat", "large"), ctx=262144),
        ModelSpec("meta-llama/llama-3.2-3b-instruct:free", ("chat", "small", "fast"), ctx=131072),
    ]

    def __init__(self, keys, **kw):
        # Optional attribution headers improve OpenRouter ranking; harmless if unset.
        extra = {"X-Title": "freelm"}
        extra.update(kw.pop("extra_headers", None) or {})
        # Free models churn constantly -> discover live by default, free-only.
        kw.setdefault("discover", True)
        kw.setdefault("discover_free_only", True)
        super().__init__(keys, extra_headers=extra, **kw)

    def rate_limit_scope(self, body: str) -> str:
        b = (body or "").lower()
        # e.g. "<model> is temporarily rate-limited upstream"
        if "rate-limited upstream" in b or "temporarily" in b:
            return "model"
        return "key"
