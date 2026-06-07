"""Build providers from environment variables.

Recognised vars (comma-separate to supply multiple keys per provider):

  OpenRouter : OPENROUTER_API_KEY   | FREELM_OPENROUTER_KEYS   (+ FREELM_OPENROUTER_TIER)
  Google     : GEMINI_API_KEY / GOOGLE_API_KEY / GOOGLE_AI_STUDIO_KEY | FREELM_GOOGLE_KEYS (+ FREELM_GOOGLE_TIER)
  NVIDIA NIM : NVIDIA_API_KEY / NIM_API_KEY | FREELM_NIM_KEYS         (+ FREELM_NIM_TIER)
"""
from __future__ import annotations

import os
from typing import List, Optional

from .errors import ConfigError
from .providers import GoogleAIStudio, NIM, OpenRouter
from .providers.base import Provider


def _split(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _first_env(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def providers_from_env() -> List[Provider]:
    provs: List[Provider] = []

    ork = _split(_first_env("OPENROUTER_API_KEY", "FREELM_OPENROUTER_KEYS"))
    if ork:
        provs.append(OpenRouter(ork, tier=os.getenv("FREELM_OPENROUTER_TIER", "free")))

    gk = _split(_first_env("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_KEY", "FREELM_GOOGLE_KEYS"))
    if gk:
        provs.append(GoogleAIStudio(gk, tier=os.getenv("FREELM_GOOGLE_TIER", "free")))

    nk = _split(_first_env("NVIDIA_API_KEY", "NIM_API_KEY", "FREELM_NIM_KEYS"))
    if nk:
        provs.append(NIM(nk, tier=os.getenv("FREELM_NIM_TIER", "free")))

    if not provs:
        raise ConfigError(
            "no provider keys found in environment. Set at least one of "
            "OPENROUTER_API_KEY, GEMINI_API_KEY, or NVIDIA_API_KEY."
        )
    return provs
