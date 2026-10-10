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
        # Free quotas are per project *per model* (about 20 requests/day each on
        # the 3.x models, checked 2026-10), so there is no useful key-wide daily
        # cap: a model's 429 benches just that model (rate_limit_scope below).
        "free": {"rpm": 15, "rpd": None},
        "tier1": {"rpm": 2000, "rpd": None},
    }

    # Verified live on the free tier 2026-10-10. Gemini 2.0 is shut down and
    # 2.5 answers new accounts with 404 "no longer available to new users", so
    # the list is 3.x plus the -latest aliases. Pro/preview models have no free
    # quota (429, limit 0) and are left out. Thinking models (3.x flash) can
    # spend a small max_tokens budget entirely on reasoning (empty text,
    # finish_reason=length): the lite models lead for `auto`, and the thinkers
    # are tagged "reasoning".
    DEFAULT_MODELS = [
        ModelSpec("gemini-3.5-flash-lite", ("chat", "fast", "small", "tools", "vision"), ctx=1048576),
        ModelSpec("gemini-3.1-flash-lite", ("chat", "fast", "small", "tools", "vision"), ctx=1048576),
        ModelSpec("gemini-flash-lite-latest", ("chat", "fast", "small", "tools", "vision"), ctx=1048576),
        ModelSpec("gemini-3.5-flash", ("chat", "large", "tools", "vision", "reasoning"), ctx=1048576),
        ModelSpec("gemini-3.7-flash", ("chat", "large", "tools", "vision", "reasoning"), ctx=1048576),
        ModelSpec("gemini-flash-latest", ("chat", "large", "tools", "vision", "reasoning"), ctx=1048576),
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
