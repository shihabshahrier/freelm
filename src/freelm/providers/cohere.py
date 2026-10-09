"""Cohere (https://cohere.com) via its OpenAI compatibility API.

Free **trial** keys (no card): 20 chat requests/minute per model and 1,000
calls/month, for non-commercial use (checked 2026-10). Production keys are
billed — use a trial key with freelm. A per-minute 429 benches just that model;
the monthly cap's 429 disables the key (``errors.classify``). Cohere rejects a
few OpenAI parameters (``n``, ``logit_bias``, ``parallel_tool_calls``, ...);
freelm fails such requests over to another provider.
"""
from __future__ import annotations

from typing import Any, Dict

from ..registry import ModelSpec
from .base import Provider


class Cohere(Provider):
    name = "cohere"
    base_url = "https://api.cohere.ai/compatibility/v1"

    # trial key: 20 RPM per model; the 1,000 calls/month cap can't be paced per day
    TIERS: Dict[str, Dict[str, Any]] = {"free": {"rpm": 20, "rpd": None}}

    # "Live" chat models on 2026-10-09 (docs.cohere.com/docs/models).
    DEFAULT_MODELS = [
        ModelSpec("command-a-plus-05-2026", ("chat", "large", "tools", "vision"), ctx=128000),
        ModelSpec("command-a-03-2025", ("chat", "large", "tools"), ctx=256000),
        ModelSpec("command-a-reasoning-08-2025", ("chat", "large", "tools", "reasoning"), ctx=256000),
        ModelSpec("command-a-vision-07-2025", ("chat", "vision"), ctx=128000),
        ModelSpec("command-r7b-12-2024", ("chat", "small", "fast", "tools"), ctx=128000),
    ]

    def rate_limit_scope(self, body: str) -> str:
        return "model"  # trial limits are per model
