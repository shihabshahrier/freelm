"""Mistral AI (https://mistral.ai) — OpenAI-compatible, free "Experiment" tier."""
from __future__ import annotations

from typing import Any, Dict

from ..registry import ModelSpec
from .base import Provider


class Mistral(Provider):
    name = "mistral"
    base_url = "https://api.mistral.ai/v1"

    # Free "Experiment" tier (verified 2026-06): 2 req/min, 500K tokens/min,
    # 1B tokens/month. The low RPM is real — add a paid tier or override rpm.
    TIERS: Dict[str, Dict[str, Any]] = {
        "free": {"rpm": 2, "rpd": None},
    }

    DEFAULT_MODELS = [
        ModelSpec("mistral-small-latest", ("chat", "large", "tools"), ctx=32000),
        ModelSpec("open-mistral-nemo", ("chat", "small", "fast"), ctx=128000),
    ]
