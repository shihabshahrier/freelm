"""Base provider. All three shipped providers speak OpenAI-compatible HTTP,
so the base does request shaping + response parsing; subclasses just declare
endpoint, auth, default free models, and tier limits.

Providers are pure (no I/O): the engine owns the httpx client and calls
``url`` / ``headers`` / ``parse_response`` on them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from .._keys import ANONYMOUS, KeyState, new_key_state
from .._types import ChatResponse, Choice, Message, Usage
from ..errors import ConfigError
from ..registry import ModelSpec, resolve_models

# ``smart`` routing: a provider with no fresh latency sample is assumed to answer
# in LATENCY_PRIOR_MS (a typical free tier), and samples older than LATENCY_TTL
# are forgotten — so unknown providers get tried and a provider that was slow
# once is reconsidered after a while.
LATENCY_PRIOR_MS = 2000.0
LATENCY_TTL = 600.0


def _as_text(v: Any) -> Any:
    """Workers AI sometimes sends a numeric token as a JSON number (``"content": 6``)."""
    return str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


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
        # model id -> monotonic ts until which it is benched (retired, or
        # throttled upstream); the router skips benched models.
        self._model_until: Dict[str, float] = {}

    # -- request shaping -------------------------------------------------
    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + self.chat_path

    def discovery_url(self) -> str:
        return self.base_url.rstrip("/") + self.models_path

    def auth_headers(self, key: str) -> Dict[str, str]:
        if key == ANONYMOUS:
            return {}  # keyless provider: no credentials at all
        return {"Authorization": f"Bearer {key}"}

    def headers(self, key: str) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        h.update(self.auth_headers(key))
        h.update(self.extra_headers)
        return h

    def adapt_payload(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Last look at the OpenAI-style request body before it is sent, for
        this provider's quirks (default: unchanged)."""
        return body

    def resolve_models(self, alias: Union[str, Sequence[str]]) -> List[str]:
        """Resolve an alias — or an ordered list of aliases (per-call fallback
        chain) — to concrete model ids, applying ``prefer`` and the free guard.

        An alias the free guard rejects is skipped; ``ConfigError`` is raised
        only when it leaves nothing to try."""
        aliases = [alias] if isinstance(alias, str) else list(alias)
        out: List[str] = []
        seen = set()
        guard: Optional[ConfigError] = None
        for a in aliases:
            try:
                ids = self._resolve_one(a)
            except ConfigError as e:
                guard = guard or e
                continue
            for mid in ids:
                if mid not in seen:
                    seen.add(mid)
                    out.append(mid)
        if not out and guard is not None:
            raise guard
        return out

    def knows_model(self, model_id: str) -> bool:
        """Is ``model_id`` in this provider's (discovered or built-in) list?"""
        return any(m.id == model_id for m in self.models)

    # -- per-model availability ------------------------------------------
    def bench_model(self, model_id: str, until: float) -> None:
        """Stop offering ``model_id`` until monotonic time ``until``."""
        self._model_until[model_id] = max(until, self._model_until.get(model_id, 0.0))

    def model_wait(self, model_id: str, now: float) -> float:
        """Seconds until ``model_id`` is offered again on this provider (0 = now)."""
        until = self._model_until.get(model_id)
        if until is None:
            return 0.0
        if now >= until:
            del self._model_until[model_id]
            return 0.0
        return until - now

    def model_ready(self, model_id: str, now: float) -> bool:
        return self.model_wait(model_id, now) == 0.0

    def _resolve_one(self, alias: str) -> List[str]:
        ids = resolve_models(self.models, alias)
        if len(ids) == 1 and ids[0] == alias:
            # exact id or passthrough — guard it, but don't reorder a direct ask
            self._check_free(alias)
            return ids
        if self.free_only:
            # an alias must never resolve to a paid model on a free-only provider
            paid = {m.id for m in self.models if not m.free}
            ids = [i for i in ids if i not in paid]
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

    # A provider that works without any key (Kilo Gateway, OVHcloud anonymous).
    keyless: bool = False

    def _check_free(self, model_id: str) -> None:
        """Guard for catalogs that mix paid and free models (``free_only=True``):
        a concrete id must be ``:free`` or listed as free. Providers whose whole
        account is free-tier keep ``free_only=False`` and skip the check."""
        if not self.free_only or model_id.endswith(":free"):
            return
        spec = next((m for m in self.models if m.id == model_id), None)
        if spec is not None and spec.free:
            return
        raise ConfigError(
            f"[{self.name}] {model_id!r} is not a free model. freelm is free-only by default — pass "
            f"{type(self).__name__}(key, free_only=False) to allow paid ids on your own account."
        )

    def rate_limit_scope(self, body: str) -> str:
        """Who does a 429 throttle? ``"key"`` (this key/account — cool it; the
        default), ``"model"`` (this key's per-model quota — Gemini, Groq) or
        ``"upstream"`` (the model, for everyone — OpenRouter's shared free pool)."""
        return "key"

    def transient_scope(self, body: str) -> str:
        """Is a 5xx/timeout about the whole provider (``"key"``, default — cool
        the key) or one overloaded model (``"model"`` — bench just the model)?"""
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
                        content=_as_text(m.get("content")),
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

    def expected_latency(self, now: float) -> float:
        """Routing estimate (ms) for the ``smart`` strategy: the average of this
        provider's fresh latency samples, else ``LATENCY_PRIOR_MS``."""
        vals = [k.ewma_latency for k in self.keys if k.ewma_latency > 0 and now - k.latency_at < LATENCY_TTL]
        return sum(vals) / len(vals) if vals else LATENCY_PRIOR_MS

    def avg_latency(self) -> float:
        vals = [k.ewma_latency for k in self.keys if k.ewma_latency > 0]
        return sum(vals) / len(vals) if vals else float("inf")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r} keys={len(self.keys)} tier={self.tier!r}>"
