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

    # Verified live on the free tier 2026-10-09. gemini-2.0-flash and
    # gemini-2.5-pro are retired (404); Pro/preview "omni"/deep-research models
    # have no free quota (429, limit 0), so they're left out. Thinking models
    # (2.5-flash, 3.x-flash) can spend a small max_tokens budget entirely on
    # reasoning (empty text, finish_reason=length) — the non-thinking lite
    # models lead for `auto`, and the thinkers are tagged "reasoning".
    DEFAULT_MODELS = [
        ModelSpec("gemini-2.5-flash-lite", ("chat", "fast", "small", "tools", "vision"), ctx=1048576),
        ModelSpec("gemini-3.1-flash-lite", ("chat", "fast", "small", "tools", "vision"), ctx=1048576),
        ModelSpec("gemini-2.5-flash", ("chat", "fast", "large", "tools", "vision", "reasoning"), ctx=1048576),
        ModelSpec("gemini-3-flash-preview", ("chat", "large", "tools", "vision", "reasoning"), ctx=1048576),
        ModelSpec("gemini-flash-lite-latest", ("chat", "fast", "small", "tools", "vision"), ctx=1048576),
    ]

    def rate_limit_scope(self, body: str) -> str:
        # AI Studio quotas (RPM/TPM/RPD) are per model, so a 429 on one model
        # leaves the others usable on the same key.
        return "model"

    def transient_scope(self, body: str) -> str:
        # "This model is currently experiencing high demand" (503) is about one
        # model; other Gemini models on the same key keep working.
        b = (body or "").lower()
        return "model" if ("high demand" in b or "overloaded" in b) else "key"


# Friendly alias
Gemini = GoogleAIStudio
