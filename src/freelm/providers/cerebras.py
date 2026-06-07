"""Cerebras (https://cerebras.ai) — OpenAI-compatible, fast inference, free tier."""
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

    DEFAULT_MODELS = [
        ModelSpec("llama-3.3-70b", ("chat", "large"), ctx=8192),
        ModelSpec("gpt-oss-120b", ("chat", "large"), ctx=8192),
        ModelSpec("qwen-3-32b", ("chat", "large"), ctx=8192),
    ]
