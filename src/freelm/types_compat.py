"""OpenAI-shaped response objects for the compat shim.

Providers already answer in the OpenAI wire format, so the shim wraps that JSON
in ``CompatObject`` — a dict with recursive attribute access — instead of
re-modelling it. Code written against the OpenAI SDK keeps working:
``r.choices[0].message.tool_calls[0].function.arguments``, ``r.usage.total_tokens``,
``r.model_dump()``, ``r.to_dict()``, ``r.model_dump_json()``. Optional SDK fields
a provider didn't send (``refusal``, ``logprobs``, ``system_fingerprint``, ...)
read as ``None``, like the SDK's pydantic models.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from ._types import ChatResponse


def _wrap(v: Any) -> Any:
    if isinstance(v, CompatObject):
        return v
    if isinstance(v, dict):
        return CompatObject(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def _plain(v: Any, exclude_none: bool) -> Any:
    if isinstance(v, dict):
        return {k: _plain(x, exclude_none) for k, x in v.items() if not (exclude_none and x is None)}
    if isinstance(v, list):
        return [_plain(x, exclude_none) for x in v]
    return v


class CompatObject(dict):
    """A dict that also reads like an OpenAI SDK model (attribute access)."""

    def __init__(self, data: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        super().__init__()
        for k, v in dict(data or {}, **kw).items():
            self[k] = _wrap(v)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.get(name)  # an optional field the provider didn't send reads as None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def model_dump(self, *, exclude_none: bool = False, **_: Any) -> Dict[str, Any]:
        return _plain(self, exclude_none)

    def to_dict(self, *, exclude_none: bool = False, **_: Any) -> Dict[str, Any]:
        return self.model_dump(exclude_none=exclude_none)

    def model_dump_json(self, *, indent: Optional[int] = None, exclude_none: bool = False, **_: Any) -> str:
        return json.dumps(self.model_dump(exclude_none=exclude_none), indent=indent)

    def to_json(self, *, indent: Optional[int] = 2, **kw: Any) -> str:
        return self.model_dump_json(indent=indent, **kw)


# Back-compat names (freelm <= 0.3 exposed separate dataclasses).
CompatCompletion = CompatObject
CompatChunk = CompatObject


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def wrap_completion(resp: ChatResponse) -> CompatObject:
    """A ``chat.completion`` object from a freelm response (provider JSON kept)."""
    raw: Dict[str, Any] = dict(resp.raw or {})
    if not raw.get("choices"):
        raw["choices"] = [
            {
                "index": c.index,
                "message": {"role": c.message.role, "content": c.message.content, "tool_calls": c.message.tool_calls},
                "finish_reason": c.finish_reason,
            }
            for c in resp.choices
        ]
    u = resp.usage
    raw["id"] = raw.get("id") or resp.id or _completion_id()
    raw["object"] = "chat.completion"
    raw["created"] = raw.get("created") or int(time.time())
    raw["model"] = raw.get("model") or resp.model
    raw["usage"] = raw.get("usage") or {
        "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens,
    }
    raw["provider"] = resp.provider  # which free tier served it (non-standard, handy)
    for c in raw["choices"]:
        msg = c.get("message") if isinstance(c, dict) else None
        if isinstance(msg, dict):
            msg.setdefault("role", "assistant")
            msg.setdefault("content", None)
            if msg.get("tool_calls") is None:
                msg.pop("tool_calls", None)  # never echo `tool_calls: null` back upstream
    return CompatObject(raw)


class _ChunkStamper:
    """Give every chunk of one stream the same id/created, as the SDK does."""

    def __init__(self) -> None:
        self.id = _completion_id()
        self.created = int(time.time())

    def __call__(self, chunk: Dict[str, Any]) -> CompatObject:
        c = dict(chunk)
        c["id"] = c.get("id") or self.id
        c["object"] = "chat.completion.chunk"
        c["created"] = c.get("created") or self.created
        c.setdefault("model", None)
        choices: List[Any] = c.get("choices") or []
        for ch in choices:
            if isinstance(ch, dict):
                ch.setdefault("index", 0)
                ch.setdefault("delta", {})
                ch.setdefault("finish_reason", None)
        c["choices"] = choices
        return CompatObject(c)


def wrap_chunk(content: str) -> CompatObject:
    """A minimal content-only ``chat.completion.chunk`` (kept for compatibility)."""
    return _ChunkStamper()({"choices": [{"index": 0, "delta": {"content": content}}]})
