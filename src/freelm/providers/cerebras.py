"""Cerebras (https://cerebras.ai) — OpenAI-compatible, very fast inference.

As of 2026-10 Cerebras has no permanently free tier: new accounts get trial
credits (payment method required). freelm still supports a key you already
have; when the credits run out requests fail with 402 and the key is
disabled — nothing is billed through freelm."""
from __future__ import annotations

from typing import Any, Dict

from ..registry import ModelSpec
from .base import Provider


class Cerebras(Provider):
    name = "cerebras"
    base_url = "https://api.cerebras.ai/v1"

    # Free tier (verified 2026-06): ~30 RPM, 1,000,000 tokens/day, context
    # capped at 8,192 (up to 128K on request). It is token-limited, not
    # request/day-limited, so rpd is None and we only pace rpm.
    TIERS: Dict[str, Dict[str, Any]] = {
        "free": {"rpm": 30, "rpd": None},
    }

    # The live catalog (2026-10-09) lists only these two; llama-3.3-70b and
    # qwen-3-32b are gone (404). Runtime discovery replaces this list.
    DEFAULT_MODELS = [
        ModelSpec("qwen-3.8-27b", ("chat", "small", "fast"), ctx=8192),
        ModelSpec("gpt-oss-120b", ("chat", "large", "reasoning"), ctx=8192),
    ]

    def __init__(self, keys, **kw):
        kw.setdefault("discover", not kw.get("models"))  # an explicit models= list wins
        super().__init__(keys, **kw)
