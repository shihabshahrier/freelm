"""Base provider. All three shipped providers speak OpenAI-compatible HTTP,
so the base does request shaping + response parsing; subclasses just declare
endpoint, auth, default free models, and tier limits.

Providers are pure (no I/O): the engine owns the httpx client and calls
``url`` / ``headers`` / ``parse_response`` on them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from .._keys import KeyState, new_key_state
from .._types import ChatResponse, Choice, Message, Usage
from ..errors import ConfigError
from ..registry import ModelSpec, resolve_models


class Provider:
    name: str = "base"
    base_url: str = ""
    chat_path: str = "/chat/completions"
    models_path: str = "/models"
    DEFAULT_MODELS: List[ModelSpec] = []
    # tier -> {"rpm": float|None, "rpd": int|None}
    TIERS: Dict[str, Dict[str, Any]] = {"free": {"rpm": 20, "rpd": None}}

    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        *,
        tier: str = "free",
        models: Optional[Sequence[ModelSpec]] = None,
        rpm: Optional[float] = None,
        rpd: Optional[int] = None,
        priority: int = 0,
        prefer: Optional[Sequence[str]] = None,
        free_only: bool = False,
        name: Optional[str] = None,
        base_url: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        discover: bool = False,
        discover_free_only: bool = False,
        cache_ttl: Optional[float] = None,
    ) -> None:
        if isinstance(keys, str):
            keys = [keys]
        keys = [k.strip() for k in (keys or []) if k and k.strip()]
        if not keys:
            raise ConfigError(f"{name or self.name}: no API keys provided")

        if name:
            self.name = name
        if base_url:
            self.base_url = base_url

        self.tier = tier
        tdef = self.TIERS.get(tier, {})
        self.rpm = rpm if rpm is not None else tdef.get("rpm", 20)
        self.rpd = rpd if rpd is not None else tdef.get("rpd", None)
        self.priority = priority
        self.prefer = list(prefer or [])
        self.free_only = free_only
        self.extra_headers = dict(extra_headers or {})
        self.discover = discover
        self.discover_free_only = discover_free_only
        self.cache_ttl = cache_ttl
        self._discovered = False
        self.models: List[ModelSpec] = list(models) if models else list(self.DEFAULT_MODELS)
        self.keys: List[KeyState] = [
            new_key_state(k, tier=tier, rpm=self.rpm, rpd=self.rpd) for k in keys
        ]
        self._rr = 0  # key round-robin cursor

    # -- request shaping -------------------------------------------------
    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + self.chat_path

    def discovery_url(self) -> str:
        return self.base_url.rstrip("/") + self.models_path

    def auth_headers(self, key: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {key}"}

    def headers(self, key: str) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        h.update(self.auth_headers(key))
        h.update(self.extra_headers)
        return h

    def resolve_models(self, alias: Union[str, Sequence[str]]) -> List[str]:
        """Resolve an alias — or an ordered list of aliases (per-call fallback
        chain) — to concrete model ids, applying ``prefer`` and the free guard."""
        aliases = [alias] if isinstance(alias, str) else list(alias)
        out: List[str] = []
        seen = set()
        for a in aliases:
            for mid in self._resolve_one(a):
                if mid not in seen:
                    seen.add(mid)
                    out.append(mid)
        return out

    def _resolve_one(self, alias: str) -> List[str]:
        ids = resolve_models(self.models, alias)
        if len(ids) == 1 and ids[0] == alias:
            # exact id or passthrough — guard it, but don't reorder a direct ask
            self._check_free(alias)
            return ids
        return self._apply_prefer(ids)

    def _apply_prefer(self, ids: List[str]) -> List[str]:
        """Move ids matching ``prefer`` patterns (exact id, else case-insensitive
        substring) to the front, in pattern order; the rest keep their order."""
        if not self.prefer:
            return ids
        front: List[str] = []
        rest = list(ids)
        for pat in self.prefer:
            low = pat.lower()
            matches = [i for i in rest if i == pat] or [i for i in rest if low in i.lower()]
            for m in matches:
                rest.remove(m)
                front.append(m)
        return front + rest

    def _check_free(self, model_id: str) -> None:
        """Hook for providers whose catalog mixes paid and free models.
        Base: every model on the account's tier is free — nothing to check."""

    def rate_limit_scope(self, body: str) -> str:
        """Is a 429 account/key-wide (``"key"``) or just this model (``"model"``)?

        Default assumes key-wide (Google/NIM bill per account). Providers like
        OpenRouter, where free models get throttled upstream per-model, override
        this so we try a different model on the same key first."""
        return "key"

    # -- response parsing ------------------------------------------------
    def parse_response(self, data: Dict[str, Any], latency_ms: float) -> ChatResponse:
        choices: List[Choice] = []
        for c in data.get("choices", []) or []:
            m = c.get("message") or {}
            choices.append(
                Choice(
                    index=c.get("index", 0),
                    message=Message(
                        role=m.get("role", "assistant"),
                        content=m.get("content"),
                        tool_calls=m.get("tool_calls"),
                    ),
                    finish_reason=c.get("finish_reason"),
                )
            )
        return ChatResponse(
            id=data.get("id"),
            model=data.get("model"),
            provider=self.name,
            choices=choices,
            usage=Usage.from_dict(data.get("usage")),
            latency_ms=latency_ms,
            raw=data,
        )

    # -- routing helpers -------------------------------------------------
    def capacity(self, now: float) -> float:
        return sum(k.remaining(now) for k in self.keys)

    def avg_latency(self) -> float:
        vals = [k.ewma_latency for k in self.keys if k.ewma_latency > 0]
        return sum(vals) / len(vals) if vals else float("inf")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r} keys={len(self.keys)} tier={self.tier!r}>"
