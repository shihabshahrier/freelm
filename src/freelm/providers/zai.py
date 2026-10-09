"""Z.ai (https://z.ai) — GLM models over an OpenAI-compatible API.

Only the *Flash* models are free (GLM-4.7-Flash, GLM-4.5-Flash, GLM-4.6V-Flash
per the official pricing page, checked 2026-10); every other GLM model is billed
against the account balance. So the provider is free-only with an explicit
list: a paid id raises ``ConfigError`` instead of spending your credit, and
discovery stays off (``/models`` leaves the free Flash models out).

The free models are limited by concurrent requests per model (GLM-4.7-Flash: one
at a time), so a 429 benches just that model for that key. GLM-4.7-Flash thinks
by default (``reasoning_content``) — give it room in ``max_tokens``.
"""
from __future__ import annotations

from typing import Any, Dict

from ..registry import ModelSpec
from .base import Provider


class ZAI(Provider):
    name = "zai"
    base_url = "https://api.z.ai/api/paas/v4"

    # No published RPM: the free models are concurrency-limited (checked 2026-10).
    TIERS: Dict[str, Dict[str, Any]] = {"free": {"rpm": 10, "rpd": None}}

    DEFAULT_MODELS = [
        ModelSpec("glm-4.7-flash", ("chat", "tools", "reasoning"), ctx=200000),
        ModelSpec("glm-4.5-flash", ("chat", "fast", "tools"), ctx=128000),
        ModelSpec("glm-4.6v-flash", ("chat", "vision", "tools"), ctx=128000),
    ]

    def __init__(self, keys, **kw: Any) -> None:
        kw.setdefault("discover", False)
        kw.setdefault("free_only", True)
        super().__init__(keys, **kw)

    def rate_limit_scope(self, body: str) -> str:
        # 1302 (this account's concurrency for the model) and 1305 (the model is
        # overloaded) are both about one model; the other models stay usable
        return "model"
