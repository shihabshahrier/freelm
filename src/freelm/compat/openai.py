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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from ..client import AsyncFreeLLM, FreeLLM
from ..types_compat import wrap_completion  # noqa: F401  (kept for re-export symmetry)


def _create_kwargs(kw: dict) -> dict:
    # OpenAI passes some args freelm handles separately or ignores.
    kw.pop("stream", None)
    return kw


class _Completions:
    def __init__(self, client: FreeLLM) -> None:
        self._client = client

    def create(self, *, model: str = "auto", messages: Optional[List[Any]] = None, **kw: Any):
        from ..types_compat import wrap_completion

        resp = self._client.chat(messages or [], model=model, **_create_kwargs(kw))
        return wrap_completion(resp)


class _AsyncCompletions:
    def __init__(self, client: AsyncFreeLLM) -> None:
        self._client = client

    async def create(self, *, model: str = "auto", messages: Optional[List[Any]] = None, **kw: Any):
        from ..types_compat import wrap_completion

        resp = await self._client.chat(messages or [], model=model, **_create_kwargs(kw))
        return wrap_completion(resp)


class _Chat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


class OpenAI:
    """Synchronous OpenAI-compatible facade backed by FreeLLM."""

    def __init__(self, freelm: Optional[FreeLLM] = None, **kw: Any) -> None:
        self._client = freelm or FreeLLM.from_env(**kw)
        self.chat = _Chat(_Completions(self._client))

    def close(self) -> None:
        self._client.close()


class AsyncOpenAI:
    """Asynchronous OpenAI-compatible facade backed by AsyncFreeLLM."""

    def __init__(self, freelm: Optional[AsyncFreeLLM] = None, **kw: Any) -> None:
        self._client = freelm or AsyncFreeLLM.from_env(**kw)
        self.chat = _Chat(_AsyncCompletions(self._client))

    async def aclose(self) -> None:
        await self._client.aclose()
