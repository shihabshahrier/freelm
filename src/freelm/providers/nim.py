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

    DEFAULT_MODELS = [
        ModelSpec("meta/llama-3.3-70b-instruct", ("chat", "large"), ctx=128000),
        ModelSpec("meta/llama-3.1-70b-instruct", ("chat", "large"), ctx=128000),
        ModelSpec("meta/llama-3.1-8b-instruct", ("chat", "small", "fast"), ctx=128000),
    ]
