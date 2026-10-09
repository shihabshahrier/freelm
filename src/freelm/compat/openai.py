"""Drop-in OpenAI-style shim.

Swap an existing OpenAI client with almost no code change::

    # from openai import OpenAI
    from freelm.compat import OpenAI

    client = OpenAI()                       # uses FreeLLM.from_env()
    r = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "hi"}],
    )
    print(r.choices[0].message.content)

OpenAI-SDK constructor arguments (``api_key``, ``base_url``, ``organization``,
...) are accepted and ignored — keys come from the environment / providers.
``stream=True`` returns an iterator of ``chat.completion.chunk`` objects
(content, tool-call deltas and finish reasons), usable as a context manager.

Tools that only speak HTTP (Cursor, Open WebUI, LangChain's ``ChatOpenAI``, ...)
should use ``freelm serve`` instead, which exposes the same router as a local
OpenAI-compatible endpoint.
"""
from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from ..client import AsyncFreeLLM, FreeLLM
from ..registry import _VIRTUAL
from ..types_compat import CompatChunk, CompatObject, _ChunkStamper, wrap_chunk, wrap_completion  # noqa: F401

# OpenAI-SDK constructor kwargs we accept for drop-in compatibility but do not
# use (freelm reads keys/endpoints from its providers). ``timeout`` is forwarded
# when it's a plain number, since the semantic matches FreeLLM's.
_OPENAI_CTOR_KWARGS = frozenset(
    {
        "api_key",
        "organization",
        "project",
        "base_url",
        "websocket_base_url",
        "max_retries",
        "default_headers",
        "default_query",
        "http_client",
        "_strict_response_validation",
    }
)
# Per-request SDK options that are about the HTTP call, not the model.
_REQUEST_ONLY = ("extra_headers", "extra_query", "timeout", "stream_options")


def _client_kwargs(kw: dict) -> dict:
    for k in _OPENAI_CTOR_KWARGS:
        kw.pop(k, None)
    t = kw.get("timeout")
    if t is not None and not isinstance(t, (int, float)):
        kw.pop("timeout")  # httpx.Timeout objects etc. — not translatable
    return kw


def _create_kwargs(kw: dict) -> dict:
    kw.pop("stream", None)
    for k in _REQUEST_ONLY:
        kw.pop(k, None)
    extra = kw.pop("extra_body", None)
    if isinstance(extra, dict):
        kw.update(extra)  # the SDK merges extra_body into the JSON body
    return {k: v for k, v in kw.items() if v is not None}


class Stream:
    """Sync stream of ``chat.completion.chunk`` objects; iterable and a context manager."""

    def __init__(self, chunks: Iterator[Dict[str, Any]]) -> None:
        self._chunks = chunks
        self._stamp = _ChunkStamper()

    def __iter__(self) -> Iterator[CompatObject]:
        for c in self._chunks:
            yield self._stamp(c)

    def __next__(self) -> CompatObject:
        return self._stamp(next(self._chunks))

    def close(self) -> None:
        close = getattr(self._chunks, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncStream:
    """Async stream of ``chat.completion.chunk`` objects; ``async for`` / ``async with``."""

    def __init__(self, chunks: AsyncIterator[Dict[str, Any]]) -> None:
        self._chunks = chunks
        self._stamp = _ChunkStamper()

    def __aiter__(self) -> "AsyncStream":
        return self

    async def __anext__(self) -> CompatObject:
        return self._stamp(await self._chunks.__anext__())

    async def close(self) -> None:
        aclose = getattr(self._chunks, "aclose", None)
        if callable(aclose):
            await aclose()

    async def __aenter__(self) -> "AsyncStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


class _Completions:
    def __init__(self, client: FreeLLM) -> None:
        self._client = client

    def create(self, *, model: str = "auto", messages: Optional[List[Any]] = None, stream: bool = False, **kw: Any):
        if stream:
            return Stream(self._client.stream_chunks(messages or [], model=model, **_create_kwargs(kw)))
        resp = self._client.chat(messages or [], model=model, **_create_kwargs(kw))
        return wrap_completion(resp)


class _AsyncCompletions:
    def __init__(self, client: AsyncFreeLLM) -> None:
        self._client = client

    async def create(self, *, model: str = "auto", messages: Optional[List[Any]] = None, stream: bool = False, **kw: Any):
        if stream:
            return AsyncStream(self._client.astream_chunks(messages or [], model=model, **_create_kwargs(kw)))
        resp = await self._client.chat(messages or [], model=model, **_create_kwargs(kw))
        return wrap_completion(resp)


class _Chat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


def _model_list(client: Any) -> CompatObject:
    """Virtual aliases first (what most callers want), then each provider's models."""
    now = int(time.time())
    data: List[Dict[str, Any]] = [
        {"id": a, "object": "model", "created": now, "owned_by": "freelm"} for a in sorted(_VIRTUAL)
    ]
    seen = {d["id"] for d in data}
    for p in client.providers:
        for m in p.models:
            if m.id not in seen:
                seen.add(m.id)
                data.append({"id": m.id, "object": "model", "created": now, "owned_by": p.name})
    return CompatObject({"object": "list", "data": data})


class _Models:
    def __init__(self, client: FreeLLM) -> None:
        self._client = client

    def list(self) -> CompatObject:
        self._client._ensure_discovered()
        return _model_list(self._client)


class _AsyncModels:
    def __init__(self, client: AsyncFreeLLM) -> None:
        self._client = client

    async def list(self) -> CompatObject:
        await self._client._ensure_discovered()
        return _model_list(self._client)


class OpenAI:
    """Synchronous OpenAI-compatible facade backed by FreeLLM."""

    def __init__(self, freelm: Optional[FreeLLM] = None, **kw: Any) -> None:
        self._client = freelm or FreeLLM.from_env(**_client_kwargs(kw))
        self.chat = _Chat(_Completions(self._client))
        self.models = _Models(self._client)

    def with_options(self, **_: Any) -> "OpenAI":
        return self  # per-request transport options don't apply to freelm

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenAI":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncOpenAI:
    """Asynchronous OpenAI-compatible facade backed by AsyncFreeLLM."""

    def __init__(self, freelm: Optional[AsyncFreeLLM] = None, **kw: Any) -> None:
        self._client = freelm or AsyncFreeLLM.from_env(**_client_kwargs(kw))
        self.chat = _Chat(_AsyncCompletions(self._client))
        self.models = _AsyncModels(self._client)

    def with_options(self, **_: Any) -> "AsyncOpenAI":
        return self

    async def close(self) -> None:
        await self._client.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncOpenAI":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
