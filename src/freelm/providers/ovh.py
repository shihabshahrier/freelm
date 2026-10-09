"""OVHcloud AI Endpoints (https://endpoints.ai.cloud.ovh.net) — OpenAI-compatible
and usable **anonymously**: 2 requests/minute per IP *per model* (verified
2026-10). An OVHcloud key switches to pay-as-you-go, so freelm only uses the
anonymous tier — a last-resort, no-signup fallback.
"""
from __future__ import annotations

from typing import Any, Dict

from .._keys import ANONYMOUS
from ..registry import ModelSpec
from .base import Provider


class OVHcloud(Provider):
    name = "ovh"
    base_url = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
    keyless = True

    # 2/min is per model; key-level pacing stays loose and each model is
    # benched on its own 429 (rate_limit_scope below).
    TIERS: Dict[str, Dict[str, Any]] = {"free": {"rpm": 10, "rpd": None}}

    # From the public catalog, 2026-10-09; discovery replaces the list.
    DEFAULT_MODELS = [
        ModelSpec("Meta-Llama-3_3-70B-Instruct", ("chat", "large")),
        ModelSpec("Qwen3.8-27B", ("chat",)),
        ModelSpec("Mistral-Small-3.2-24B-Instruct-2506", ("chat",)),
        ModelSpec("gpt-oss-120b", ("chat", "large", "reasoning")),
        ModelSpec("gpt-oss-20b", ("chat", "small", "fast", "reasoning")),
        ModelSpec("Qwen3.5-9B", ("chat", "small", "fast")),
    ]

    def __init__(self, **kw: Any) -> None:
        kw.pop("keys", None)  # keys mean paid usage here — anonymous only
        kw.setdefault("discover", not kw.get("models"))
        super().__init__(ANONYMOUS, **kw)

    def rate_limit_scope(self, body: str) -> str:
        return "model"  # the anonymous limit is per IP *per model*
