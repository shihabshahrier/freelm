"""Kilo Gateway (https://kilo.ai) — OpenAI-compatible gateway whose free routes
work **without any key** (about 200 requests/hour per IP, verified 2026-10); a
free Kilo account key lifts that limit. Its catalog mixes paid and free models,
so like OpenRouter it is free-only by default. Free routes may log prompts —
the catalog's ``mayTrainOnYourPrompts`` flag says which.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

from .._keys import ANONYMOUS
from ..registry import ModelSpec
from .base import Provider


class Kilo(Provider):
    name = "kilo"
    base_url = "https://api.kilo.ai/api/gateway"
    keyless = True

    # anonymous: ~200 requests/hour per IP -> pace to 3/min
    TIERS: Dict[str, Dict[str, Any]] = {"free": {"rpm": 3, "rpd": None}}

    # Free routes answering keyless on 2026-10-09; discovery replaces the list.
    DEFAULT_MODELS = [
        ModelSpec("poolside/laguna-s-2.1:free", ("chat", "tools"), ctx=262144),
        ModelSpec("nvidia/nemotron-3-super-120b-a12b:free", ("chat", "large", "tools", "reasoning"), ctx=262144),
        ModelSpec("kilo-auto/free", ("chat", "tools", "reasoning"), ctx=256000),
        ModelSpec("openrouter/free", ("chat",), ctx=200000),
    ]

    def __init__(self, keys: Optional[Union[str, Sequence[str]]] = None, **kw: Any) -> None:
        kw.setdefault("discover", not kw.get("models"))  # an explicit models= list wins
        kw.setdefault("discover_free_only", True)
        kw.setdefault("free_only", True)
        super().__init__(keys or ANONYMOUS, **kw)
