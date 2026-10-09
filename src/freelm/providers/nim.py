"""NVIDIA NIM (https://build.nvidia.com) — OpenAI-compatible.

Base: https://integrate.api.nvidia.com/v1
Auth: Authorization: Bearer nvapi-...

Free usage is metered against build.nvidia.com credits; there is no fixed public
rpd, so ``rpd`` defaults to None (unlimited until credits run out) and we just
pace rpm.
"""
from __future__ import annotations

from typing import Any, Dict

from ..registry import ModelSpec
from .base import Provider


class NIM(Provider):
    name = "nim"
    base_url = "https://integrate.api.nvidia.com/v1"

    TIERS: Dict[str, Dict[str, Any]] = {
        "free": {"rpm": 40, "rpd": None},
    }

    # meta/llama-3.x reached end of life 2026-08-26 (HTTP 410). These ids are
    # in the live catalog as of 2026-10-09; retired ones get benched at runtime.
    DEFAULT_MODELS = [
        ModelSpec("nvidia/nemotron-3-super-120b-a12b", ("chat", "large", "tools", "reasoning")),
        ModelSpec("deepseek-ai/deepseek-v4.1-flash", ("chat", "large", "tools")),
        ModelSpec("z-ai/glm-5.3-flash", ("chat", "fast", "tools")),
        ModelSpec("moonshotai/kimi-k2.6", ("chat", "large", "tools")),
        ModelSpec("nvidia/nemotron-3.5-lightning-30b-a3b", ("chat", "small", "fast")),
        ModelSpec("openai/gpt-oss-20b", ("chat", "small", "fast", "tools", "reasoning")),
    ]
