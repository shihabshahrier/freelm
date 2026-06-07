"""Groq (https://groq.com) — OpenAI-compatible, very fast inference, free dev tier."""
from __future__ import annotations

from typing import Any, Dict

from ..registry import ModelSpec
from .base import Provider


class Groq(Provider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"

    # Free tier (verified 2026-06): 30 RPM, 6K TPM, 14,400 req/day, resets
    # midnight UTC. Limits apply org-wide. No credit card required.
    TIERS: Dict[str, Dict[str, Any]] = {
        "free": {"rpm": 30, "rpd": 14400},
    }

    # Verified exact IDs (console.groq.com/docs/models, 2026-06).
    DEFAULT_MODELS = [
        ModelSpec("llama-3.3-70b-versatile", ("chat", "large", "tools"), ctx=128000),
        ModelSpec("openai/gpt-oss-120b", ("chat", "large", "tools"), ctx=131072),
        ModelSpec("openai/gpt-oss-20b", ("chat", "small", "fast", "tools"), ctx=131072),
        ModelSpec("llama-3.1-8b-instant", ("chat", "small", "fast"), ctx=128000),
    ]

    def __init__(self, keys, **kw):
        # Self-correct model list from the live /models endpoint (non-chat
        # models like whisper are filtered out in discovery).
        kw.setdefault("discover", True)
        super().__init__(keys, **kw)
