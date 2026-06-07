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

    DEFAULT_MODELS = [
        ModelSpec("llama-3.3-70b-versatile", ("chat", "large", "tools"), ctx=128000),
        ModelSpec("openai/gpt-oss-20b", ("chat", "small", "fast", "tools"), ctx=131072),
        ModelSpec("llama-3.1-8b-instant", ("chat", "small", "fast"), ctx=128000),
    ]
