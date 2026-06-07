"""Google AI Studio (Gemini) via its OpenAI-compatible endpoint.

Base: https://generativelanguage.googleapis.com/v1beta/openai
Auth: Authorization: Bearer <AI Studio API key>

Free tier rpm/rpd are per-model and change often; values below are conservative
defaults for flash-class models. Tier 1 (billing enabled) lifts them sharply.
"""
from __future__ import annotations

from typing import Any, Dict

from ..registry import ModelSpec
from .base import Provider


class GoogleAIStudio(Provider):
    name = "google"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai"

    TIERS: Dict[str, Dict[str, Any]] = {
        "free": {"rpm": 15, "rpd": 1500},
        "tier1": {"rpm": 2000, "rpd": None},
    }

    DEFAULT_MODELS = [
        ModelSpec("gemini-2.0-flash", ("chat", "fast", "large"), ctx=1000000),
        ModelSpec("gemini-2.0-flash-lite", ("chat", "fast", "small"), ctx=1000000),
        ModelSpec("gemini-1.5-flash", ("chat", "fast"), ctx=1000000),
    ]


# Friendly alias
Gemini = GoogleAIStudio
