"""Groq (https://groq.com) — OpenAI-compatible, very fast inference, free dev tier."""
from __future__ import annotations

from typing import Any, Dict

from ..registry import ModelSpec
from .base import Provider


class Groq(Provider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"

    # Free tier (console.groq.com/docs/rate-limits, checked 2026-10): per model
    # 30 RPM, 1K RPD, 8K TPM, 200K TPD. No credit card required.
    TIERS: Dict[str, Dict[str, Any]] = {
        "free": {"rpm": 30, "rpd": 1000},
    }

    # llama-3.3-70b-versatile and llama-3.1-8b-instant were shut down for free/dev
    # accounts on 2026-08-16; these are Groq's named successors (deprecations
    # page, checked 2026-10). Live discovery replaces this list at runtime.
    DEFAULT_MODELS = [
        ModelSpec("qwen/qwen3.6-27b", ("chat", "tools")),
        ModelSpec("openai/gpt-oss-120b", ("chat", "large", "tools", "reasoning"), ctx=131072),
        ModelSpec("openai/gpt-oss-20b", ("chat", "small", "fast", "tools", "reasoning"), ctx=131072),
    ]

    def __init__(self, keys, **kw):
        # Self-correct model list from the live /models endpoint (non-chat
        # models like whisper are filtered out in discovery) — unless the
        # caller pinned an explicit models= list.
        kw.setdefault("discover", not kw.get("models"))
        super().__init__(keys, **kw)

    def rate_limit_scope(self, body: str) -> str:
        # Groq limits (RPM/RPD/TPM/TPD) apply per model, so a 429 on one model
        # leaves the others usable on the same key.
        return "model"
