"""Dynamic model discovery via the OpenAI-compatible ``GET /models`` endpoint.

Free-tier model ids churn constantly, so the registry hardcoded in each provider
is only a *fallback*. When a provider has ``discover=True`` (OpenRouter by
default), freelm queries ``/models`` at first use, derives tags from metadata,
caches the raw list to disk (TTL), and replaces the provider's model list.

Resolution order: live API -> disk cache -> hardcoded defaults.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import httpx

from . import _cache
from .registry import ModelSpec

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b")
# substrings that mark a non-chat model — some providers' /models list audio,
# embedding, etc. with no modality metadata, so filter them out by name.
_NON_CHAT = (
    "whisper", "tts", "text-to-speech", "speech", "audio", "transcribe",
    "embed", "embedding", "rerank", "moderation", "guard", "ocr", "-vision-encoder",
    "imagen", "veo", "image-generation", "-generate-", "stable-diffusion", "dall-e", "aqa",
    "orpheus", "playai", "sonic", "voice", "-stt", "-asr",
    # classifiers / safety filters / reward models answer with labels, not chat
    "safety", "content-safety", "topic-control", "reward", "-parse", "nemoretriever", "nvclip", "bge",
)
# size words are matched as whole tokens of the id ("gemini" must not read as "mini")
_LARGE_HINTS = frozenset({"ultra", "super", "large", "xl", "405b", "235b", "120b"})
_SMALL_HINTS = frozenset({"mini", "nano", "small", "lite", "tiny", "xs", "edge"})
_TOKEN_SPLIT = re.compile(r"[^a-z0-9.]+")
# reasoning models emit verbose hidden thinking — detect by name when /models
# carries no metadata, so `auto` can deprioritize them.
_REASONING_HINTS = ("gpt-oss", "deepseek-r1", "magistral", "qwq", "thinking", "-think", "reasoning")


def _size_tags(model_id: str) -> List[str]:
    s = model_id.lower()
    tokens = set(_TOKEN_SPLIT.split(s))
    small_kw = bool(tokens & _SMALL_HINTS)
    large_kw = bool(tokens & _LARGE_HINTS)
    # explicit keyword (nano/mini/ultra/...) wins over raw parameter count,
    # which can be misleading (e.g. "nano-30b" is meant to be the small one).
    if small_kw and not large_kw:
        return ["small", "fast"]
    if large_kw and not small_kw:
        return ["large"]
    nums = [float(x) for x in _SIZE_RE.findall(s)]
    big = max(nums) if nums else None
    if big is not None:
        if big >= 30:
            return ["large"]
        if big <= 20:
            return ["small", "fast"]
    return []


def _is_free(mid: str, m: Dict[str, Any], *, strict: bool = False) -> bool:
    """``:free`` ids, entries flagged ``isFree`` (Kilo) and zero-priced entries are
    free. Catalogs without pricing (Groq, Mistral, NIM, ...) belong to free-tier
    accounts, so they count as free — except under ``strict`` (filtering a mixed
    paid/free catalog), where only positive evidence does."""
    if mid.endswith(":free"):
        return True
    if isinstance(m.get("isFree"), bool):
        return m["isFree"]
    pricing = m.get("pricing")
    if not isinstance(pricing, dict) or not pricing:
        return not strict
    try:
        # every listed price must be zero (image/audio/request fees included)
        return all(float(v) == 0 for v in pricing.values() if v is not None and not isinstance(v, bool))
    except (TypeError, ValueError):
        return False


def to_specs(api_models: List[Dict[str, Any]], *, free_only: bool) -> List[ModelSpec]:
    """Normalize raw OpenAI-/OpenRouter-style model dicts into ordered ModelSpecs."""
    specs: List[ModelSpec] = []
    for m in api_models:
        mid = m.get("id")
        if not mid or not isinstance(mid, str):
            continue
        if mid.startswith("models/"):
            mid = mid[len("models/"):]  # Google lists "models/gemini-..."; chat takes the bare id
        if free_only and not _is_free(mid, m, strict=True):
            continue  # paid entry in a mixed catalog (OpenRouter, Kilo)
        if any(t in mid.lower() for t in _NON_CHAT):
            continue  # audio / embedding / etc. — not a chat model

        arch = m.get("architecture") or {}
        out_mod = m.get("output_modalities") or arch.get("output_modalities") or ["text"]
        if set(out_mod) != {"text"}:
            continue  # audio/image/music generators (even "text+audio" ones) aren't chat models
        # Mistral: {"capabilities": {"completion_chat": true, "function_calling": true, "vision": false}}
        caps = m.get("capabilities") if isinstance(m.get("capabilities"), dict) else {}
        if caps.get("completion_chat") is False:
            continue

        ctx = (
            m.get("context_length")
            or m.get("context_window")  # Groq
            or m.get("max_context_length")  # Mistral
            or (m.get("top_provider") or {}).get("context_length")
            or 0
        )
        params = [str(p).lower() for p in (m.get("supported_parameters") or [])]
        in_mod = arch.get("input_modalities") or []

        tags = ["chat"] + _size_tags(mid)
        if "tools" in params or "tool_choice" in params or caps.get("function_calling"):
            tags.append("tools")
        low = mid.lower()
        if "reasoning" in params or "include_reasoning" in params or any(h in low for h in _REASONING_HINTS):
            tags.append("reasoning")
        if "image" in in_mod or "vision" in params or caps.get("vision"):
            tags.append("vision")

        # dedupe, keep order
        tags = list(dict.fromkeys(tags))
        try:
            ctx = int(ctx)
        except (TypeError, ValueError):
            ctx = 0
        specs.append(ModelSpec(mid, tuple(tags), ctx=ctx, free=_is_free(mid, m)))

    # Default order for `auto`: capable, fast, and predictable. Giant models
    # (>150B params) are slow; reasoning models emit verbose thinking — both rank
    # after plain instruct models. Then prefer large, then bigger context window.
    # (Target a reasoning/giant model explicitly via model="vendor/id" when wanted.)
    def _params_b(mid: str) -> float:
        nums = [float(x) for x in _SIZE_RE.findall(mid.lower())]
        return max(nums) if nums else 0.0

    specs.sort(
        key=lambda s: (
            1 if _params_b(s.id) > 150 else 0,   # giant -> later
            1 if "reasoning" in s.tags else 0,   # reasoning -> later
            0 if "large" in s.tags else 1,       # prefer large
            -s.ctx,                              # bigger context first
        )
    )
    return specs


def _raw_models(provider: Any, payload: Any) -> List[Dict[str, Any]]:
    data = (payload.get("data") or payload.get("models")) if isinstance(payload, dict) else None
    return [m for m in data if isinstance(m, dict)] if isinstance(data, list) else []


def _apply(provider: Any, raw: List[Dict[str, Any]]) -> bool:
    specs = to_specs(raw, free_only=getattr(provider, "discover_free_only", False))
    if specs:
        provider.models = specs
        provider._discovered = True
        return True
    return False


# A catalog fetch must never eat the caller's whole request budget.
_DISCOVERY_TIMEOUT = 10.0


def _discovery_keys(provider: Any) -> List[str]:
    """Keys to try for ``GET /models``: usable ones first, at most three. A dead
    first key (401) must not hide a healthy second key's catalog."""
    keys = [k for k in provider.keys if not k.disabled] or list(provider.keys)
    return [k.key for k in keys[:3]]


def _accept(provider: Any, r: httpx.Response) -> Optional[bool]:
    """Handle one ``/models`` response: True/False when settled, None to try the next key."""
    if r.status_code != 200:
        return None
    raw = _raw_models(provider, r.json())
    if not raw:
        return False
    _cache.save(provider.name, raw, getattr(provider, "cache_ttl", None))
    return _apply(provider, raw)


def discover_sync(provider: Any, client: Optional[httpx.Client] = None) -> bool:
    """Populate ``provider.models`` from the live API (or cache). Returns success.
    On any failure the provider keeps its existing (hardcoded) models."""
    cached = _cache.load(provider.name)
    if cached is not None and _apply(provider, cached):
        return True  # a cached list that yields no usable specs falls through to a live fetch

    owns = client is None
    client = client or httpx.Client(timeout=_DISCOVERY_TIMEOUT)
    try:
        for key in _discovery_keys(provider):
            try:
                r = client.get(provider.discovery_url(), headers=provider.headers(key), timeout=_DISCOVERY_TIMEOUT)
                done = _accept(provider, r)
            except (httpx.HTTPError, ValueError):
                return False  # network trouble: don't hammer it once per key
            if done is not None:
                return done
    finally:
        if owns:
            client.close()
    return False


async def discover_async(provider: Any, client: httpx.AsyncClient) -> bool:
    cached = _cache.load(provider.name)
    if cached is not None and _apply(provider, cached):
        return True
    for key in _discovery_keys(provider):
        try:
            r = await client.get(provider.discovery_url(), headers=provider.headers(key), timeout=_DISCOVERY_TIMEOUT)
            done = _accept(provider, r)
        except (httpx.HTTPError, ValueError):
            return False
        if done is not None:
            return done
    return False


def list_free_models(api_key: Optional[str] = None, *, refresh: bool = False) -> List[ModelSpec]:
    """Convenience: discover OpenRouter free models without building a client.

    >>> from freelm import list_free_models
    >>> [m.id for m in list_free_models()][:3]
    """
    import os

    from .providers.openrouter import OpenRouter

    key = api_key or os.getenv("OPENROUTER_API_KEY") or "none"
    if refresh:
        _cache.clear("openrouter")
    p = OpenRouter(key)
    discover_sync(p)
    return p.models
