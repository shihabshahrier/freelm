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
``stream=True`` returns an iterator of ``chat.completion.chunk``-shaped objects.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, List, Optional

from ..client import AsyncFreeLLM, FreeLLM
from ..types_compat import CompatChunk, wrap_chunk, wrap_completion  # noqa: F401  (re-export symmetry)

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


def _client_kwargs(kw: dict) -> dict:
    for k in _OPENAI_CTOR_KWARGS:
        kw.pop(k, None)
    t = kw.get("timeout")
    if t is not None and not isinstance(t, (int, float)):
        kw.pop("timeout")  # httpx.Timeout objects etc. — not translatable
    return kw


def _create_kwargs(kw: dict) -> dict:
    # OpenAI passes some args freelm handles separately or ignores.
    kw.pop("stream", None)
    kw.pop("stream_options", None)
    return kw


class _Completions:
    def __init__(self, client: FreeLLM) -> None:
        self._client = client

    def create(self, *, model: str = "auto", messages: Optional[List[Any]] = None, stream: bool = False, **kw: Any):
        if stream:
            return self._create_stream(model, messages or [], _create_kwargs(kw))
        resp = self._client.chat(messages or [], model=model, **_create_kwargs(kw))
        return wrap_completion(resp)

    def _create_stream(self, model: str, messages: List[Any], kw: dict) -> Iterator[CompatChunk]:
        for delta in self._client.stream(messages, model=model, **kw):
            yield wrap_chunk(delta)


class _AsyncCompletions:
    def __init__(self, client: AsyncFreeLLM) -> None:
        self._client = client

    async def create(self, *, model: str = "auto", messages: Optional[List[Any]] = None, stream: bool = False, **kw: Any):
        if stream:
            return self._create_stream(model, messages or [], _create_kwargs(kw))
        resp = await self._client.chat(messages or [], model=model, **_create_kwargs(kw))
        return wrap_completion(resp)

    async def _create_stream(self, model: str, messages: List[Any], kw: dict) -> AsyncIterator[CompatChunk]:
        async for delta in self._client.astream(messages, model=model, **kw):
            yield wrap_chunk(delta)


class _Chat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


class OpenAI:
    """Synchronous OpenAI-compatible facade backed by FreeLLM."""

    def __init__(self, freelm: Optional[FreeLLM] = None, **kw: Any) -> None:
        self._client = freelm or FreeLLM.from_env(**_client_kwargs(kw))
        self.chat = _Chat(_Completions(self._client))

    def close(self) -> None:
        self._client.close()


class AsyncOpenAI:
    """Asynchronous OpenAI-compatible facade backed by AsyncFreeLLM."""

    def __init__(self, freelm: Optional[AsyncFreeLLM] = None, **kw: Any) -> None:
        self._client = freelm or AsyncFreeLLM.from_env(**_client_kwargs(kw))
        self.chat = _Chat(_AsyncCompletions(self._client))

    async def aclose(self) -> None:
        await self._client.aclose()
