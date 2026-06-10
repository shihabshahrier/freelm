"""The user-facing clients: ``FreeLLM`` (sync) and ``AsyncFreeLLM`` (async)."""
from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Sequence, Union

import httpx

from . import _engine as engine
from . import discovery
from ._state import StateStore
from ._types import ChatResponse, Event, build_request
from ._version import __version__
from .errors import ConfigError, NoProvidersAvailable, ProviderError, Transient
from .providers.base import Provider
from .strategy import Candidate, STRATEGIES

_DEFAULT_UA = f"freelm/{__version__}"

ModelArg = Union[str, Sequence[str]]


def _sse_delta(line: str) -> Optional[str]:
    """Extract the content delta from one OpenAI-style SSE line, or None."""
    if not line or not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        obj = json.loads(data)
        return obj["choices"][0]["delta"].get("content") or None
    except (ValueError, KeyError, IndexError, TypeError):
        return None


class _BaseClient:
    def __init__(
        self,
        providers: Sequence[Provider],
        *,
        strategy: str = "priority",
        max_attempts: int = 12,
        timeout: float = 60.0,
        wait: bool = False,
        max_wait: float = 20.0,
        on_event: Optional[Callable[[Event], Any]] = None,
        persist: Optional[bool] = None,
    ) -> None:
        providers = list(providers)
        if not providers:
            raise ConfigError("FreeLLM needs at least one provider")
        if strategy not in STRATEGIES:
            raise ConfigError(f"unknown strategy {strategy!r}; pick one of {STRATEGIES}")
        self.providers = providers
        self.strategy = strategy
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.wait = wait
        self.max_wait = max_wait
        self._rr: Dict[str, int] = {"p": 0}
        self._discovery_done = False
        self._on_event = on_event
        if persist is None:
            persist = os.getenv("FREELM_PERSIST", "").lower() in ("1", "true", "yes")
        self._state: Optional[StateStore] = StateStore() if persist else None
        if self._state is not None:
            self._state.load_into(self.providers, time.monotonic())

    # -- observability / persistence --------------------------------------
    def _emit(
        self,
        kind: str,
        *,
        cand: Optional[Candidate] = None,
        provider: Optional[str] = None,
        status: Optional[int] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
        attempt: int = 0,
    ) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(
                Event(
                    kind=kind,
                    provider=cand.provider.name if cand else provider,
                    key=cand.key.masked() if cand else None,
                    model=cand.model if cand else None,
                    status=status,
                    latency_ms=latency_ms,
                    error=error,
                    attempt=attempt,
                )
            )
        except Exception:
            pass  # a misbehaving callback must never break the call

    def _save_state(self) -> None:
        if self._state is not None:
            self._state.save(self.providers, time.monotonic())

    # -- introspection ---------------------------------------------------
    def health(self) -> List[Dict[str, Any]]:
        now = time.monotonic()
        out: List[Dict[str, Any]] = []
        for p in self.providers:
            for k in p.keys:
                out.append(
                    {
                        "provider": p.name,
                        "key": k.masked(),
                        "tier": k.tier,
                        "ready": k.ready(now),
                        "disabled": k.disabled,
                        "breaker": k.breaker.state,
                        "rpd_used": k.rpd_used,
                        "rpd": k.rpd,
                        "last_error": k.last_error,
                        "ewma_latency_ms": round(k.ewma_latency, 1),
                    }
                )
        return out


class FreeLLM(_BaseClient):
    """Synchronous always-up chat client."""

    def __init__(self, providers: Sequence[Provider], *, http_client: Optional[httpx.Client] = None, **kw: Any) -> None:
        super().__init__(providers, **kw)
        self._client = http_client or httpx.Client(timeout=self.timeout, headers={"User-Agent": _DEFAULT_UA})
        self._owns_client = http_client is None

    @classmethod
    def from_env(cls, **kw: Any) -> "FreeLLM":
        from .config import providers_from_env

        return cls(providers_from_env(), **kw)

    def _ensure_discovered(self) -> None:
        if self._discovery_done:
            return
        for p in self.providers:
            if getattr(p, "discover", False):
                try:
                    if discovery.discover_sync(p, self._client):
                        self._emit("discovery", provider=p.name)
                except Exception:
                    pass  # keep hardcoded fallback models
        self._discovery_done = True

    def refresh_models(self) -> None:
        """Force re-discovery on next call (bypasses the in-memory guard)."""
        self._discovery_done = False
        for p in self.providers:
            p._discovered = False

    def chat(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> ChatResponse:
        self._ensure_discovered()
        req = build_request(messages, model, kw)
        deadline = time.monotonic() + self.timeout if self.timeout else None
        attempts: List = []
        tried: set = set()

        while len(attempts) < self.max_attempts:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            cand = engine.select_candidate(self.providers, self.strategy, self._rr, req.model, tried, now)
            if cand is None:
                w = engine.soonest_wait(self.providers, now)
                if self.wait and w is not None and w <= self.max_wait and (deadline is None or now + w < deadline):
                    self._emit("wait", latency_ms=w * 1000.0, attempt=len(attempts))
                    time.sleep(min(w, self.max_wait) + 0.01)
                    tried = engine.forget_recovered(self.providers, tried, time.monotonic())
                    continue
                break

            tried.add((cand.provider.name, cand.key.key, cand.model))
            if not cand.key.reserve(now):
                continue  # lost an rpm token to a concurrent caller; pick another

            self._emit("attempt", cand=cand, attempt=len(attempts) + 1)
            try:
                resp = self._do(cand, req)
            except ProviderError as exc:
                engine.apply_error(cand, exc, time.monotonic())
                attempts.append((cand, exc))
                self._emit("error", cand=cand, status=exc.status, error=str(exc), attempt=len(attempts))
                self._save_state()
                if engine.should_raise(exc):
                    raise
                continue
            engine.apply_success(cand, resp.latency_ms)
            self._emit("success", cand=cand, latency_ms=resp.latency_ms, attempt=len(attempts) + 1)
            self._save_state()
            return resp

        raise NoProvidersAvailable(attempts)

    def text(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> str:
        return self.chat(messages, model=model, **kw).text

    def stream(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> Iterator[str]:
        """Yield content deltas as they arrive. Fails over between providers
        *before* the first token; once tokens start flowing it stays on that
        provider (no mid-stream failover)."""
        self._ensure_discovered()
        req = build_request(messages, model, kw)
        deadline = time.monotonic() + self.timeout if self.timeout else None
        attempts: List = []
        tried: set = set()

        while len(attempts) < self.max_attempts:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            cand = engine.select_candidate(self.providers, self.strategy, self._rr, req.model, tried, now)
            if cand is None:
                w = engine.soonest_wait(self.providers, now)
                if self.wait and w is not None and w <= self.max_wait and (deadline is None or now + w < deadline):
                    self._emit("wait", latency_ms=w * 1000.0, attempt=len(attempts))
                    time.sleep(min(w, self.max_wait) + 0.01)
                    tried = engine.forget_recovered(self.providers, tried, time.monotonic())
                    continue
                break

            tried.add((cand.provider.name, cand.key.key, cand.model))
            if not cand.key.reserve(now):
                continue

            produced = False
            first_ms = 0.0  # time-to-first-token; feeds the latency EWMA
            t0 = time.monotonic()
            self._emit("attempt", cand=cand, attempt=len(attempts) + 1)
            try:
                for chunk in self._stream_do(cand, req):
                    if not produced:
                        first_ms = (time.monotonic() - t0) * 1000.0
                    produced = True
                    yield chunk
            except ProviderError as exc:
                engine.apply_error(cand, exc, time.monotonic())
                attempts.append((cand, exc))
                self._emit("error", cand=cand, status=exc.status, error=str(exc), attempt=len(attempts))
                self._save_state()
                if produced or engine.should_raise(exc):
                    raise
                continue
            engine.apply_success(cand, first_ms)
            self._emit("success", cand=cand, latency_ms=first_ms, attempt=len(attempts) + 1)
            self._save_state()
            return

        raise NoProvidersAvailable(attempts)

    def _stream_do(self, cand: Candidate, req) -> Iterator[str]:
        p = cand.provider
        body = req.payload(cand.model)
        body["stream"] = True
        try:
            with self._client.stream("POST", p.url, headers=p.headers(cand.key.key), json=body) as r:
                if r.status_code != 200:
                    r.read()
                    from .errors import RateLimited, classify

                    err = classify(r.status_code, dict(r.headers), r.text, p.name)
                    if isinstance(err, RateLimited):
                        err.scope = p.rate_limit_scope(r.text)
                    raise err
                for line in r.iter_lines():
                    delta = _sse_delta(line)
                    if delta:
                        yield delta
        except httpx.TimeoutException as e:
            raise Transient(p.name, 0, f"timeout: {e}")
        except httpx.TransportError as e:
            raise Transient(p.name, 0, f"transport: {e}")

    def _do(self, cand: Candidate, req) -> ChatResponse:
        p = cand.provider
        body = req.payload(cand.model)
        t0 = time.monotonic()
        try:
            r = self._client.post(p.url, headers=p.headers(cand.key.key), json=body)
        except httpx.TimeoutException as e:
            raise Transient(p.name, 0, f"timeout: {e}")
        except httpx.TransportError as e:
            raise Transient(p.name, 0, f"transport: {e}")
        dt = (time.monotonic() - t0) * 1000.0
        if r.status_code == 200:
            return p.parse_response(r.json(), dt)
        from .errors import RateLimited, classify

        err = classify(r.status_code, dict(r.headers), r.text, p.name)
        if isinstance(err, RateLimited):
            err.scope = p.rate_limit_scope(r.text)
        raise err

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        self._save_state()
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "FreeLLM":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncFreeLLM(_BaseClient):
    """Asynchronous always-up chat client."""

    def __init__(self, providers: Sequence[Provider], *, http_client: Optional[httpx.AsyncClient] = None, **kw: Any) -> None:
        super().__init__(providers, **kw)
        self._client = http_client
        self._owns_client = http_client is None

    @classmethod
    def from_env(cls, **kw: Any) -> "AsyncFreeLLM":
        from .config import providers_from_env

        return cls(providers_from_env(), **kw)

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, headers={"User-Agent": _DEFAULT_UA})
        return self._client

    async def _ensure_discovered(self) -> None:
        if self._discovery_done:
            return
        client = self._ensure_client()
        for p in self.providers:
            if getattr(p, "discover", False):
                try:
                    if await discovery.discover_async(p, client):
                        self._emit("discovery", provider=p.name)
                except Exception:
                    pass  # keep hardcoded fallback models
        self._discovery_done = True

    def refresh_models(self) -> None:
        self._discovery_done = False
        for p in self.providers:
            p._discovered = False

    async def chat(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> ChatResponse:
        import asyncio

        await self._ensure_discovered()
        req = build_request(messages, model, kw)
        deadline = time.monotonic() + self.timeout if self.timeout else None
        attempts: List = []
        tried: set = set()

        while len(attempts) < self.max_attempts:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            cand = engine.select_candidate(self.providers, self.strategy, self._rr, req.model, tried, now)
            if cand is None:
                w = engine.soonest_wait(self.providers, now)
                if self.wait and w is not None and w <= self.max_wait and (deadline is None or now + w < deadline):
                    self._emit("wait", latency_ms=w * 1000.0, attempt=len(attempts))
                    await asyncio.sleep(min(w, self.max_wait) + 0.01)
                    tried = engine.forget_recovered(self.providers, tried, time.monotonic())
                    continue
                break

            tried.add((cand.provider.name, cand.key.key, cand.model))
            if not cand.key.reserve(now):
                continue

            self._emit("attempt", cand=cand, attempt=len(attempts) + 1)
            try:
                resp = await self._ado(cand, req)
            except ProviderError as exc:
                engine.apply_error(cand, exc, time.monotonic())
                attempts.append((cand, exc))
                self._emit("error", cand=cand, status=exc.status, error=str(exc), attempt=len(attempts))
                self._save_state()
                if engine.should_raise(exc):
                    raise
                continue
            engine.apply_success(cand, resp.latency_ms)
            self._emit("success", cand=cand, latency_ms=resp.latency_ms, attempt=len(attempts) + 1)
            self._save_state()
            return resp

        raise NoProvidersAvailable(attempts)

    async def text(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> str:
        return (await self.chat(messages, model=model, **kw)).text

    async def astream(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> AsyncIterator[str]:
        """Async content-delta stream. Fails over before the first token only."""
        import asyncio

        await self._ensure_discovered()
        req = build_request(messages, model, kw)
        deadline = time.monotonic() + self.timeout if self.timeout else None
        attempts: List = []
        tried: set = set()

        while len(attempts) < self.max_attempts:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            cand = engine.select_candidate(self.providers, self.strategy, self._rr, req.model, tried, now)
            if cand is None:
                w = engine.soonest_wait(self.providers, now)
                if self.wait and w is not None and w <= self.max_wait and (deadline is None or now + w < deadline):
                    self._emit("wait", latency_ms=w * 1000.0, attempt=len(attempts))
                    await asyncio.sleep(min(w, self.max_wait) + 0.01)
                    tried = engine.forget_recovered(self.providers, tried, time.monotonic())
                    continue
                break

            tried.add((cand.provider.name, cand.key.key, cand.model))
            if not cand.key.reserve(now):
                continue

            produced = False
            first_ms = 0.0  # time-to-first-token; feeds the latency EWMA
            t0 = time.monotonic()
            self._emit("attempt", cand=cand, attempt=len(attempts) + 1)
            try:
                async for chunk in self._astream_do(cand, req):
                    if not produced:
                        first_ms = (time.monotonic() - t0) * 1000.0
                    produced = True
                    yield chunk
            except ProviderError as exc:
                engine.apply_error(cand, exc, time.monotonic())
                attempts.append((cand, exc))
                self._emit("error", cand=cand, status=exc.status, error=str(exc), attempt=len(attempts))
                self._save_state()
                if produced or engine.should_raise(exc):
                    raise
                continue
            engine.apply_success(cand, first_ms)
            self._emit("success", cand=cand, latency_ms=first_ms, attempt=len(attempts) + 1)
            self._save_state()
            return

        raise NoProvidersAvailable(attempts)

    async def _astream_do(self, cand: Candidate, req) -> AsyncIterator[str]:
        p = cand.provider
        client = self._ensure_client()
        body = req.payload(cand.model)
        body["stream"] = True
        try:
            async with client.stream("POST", p.url, headers=p.headers(cand.key.key), json=body) as r:
                if r.status_code != 200:
                    await r.aread()
                    from .errors import RateLimited, classify

                    err = classify(r.status_code, dict(r.headers), r.text, p.name)
                    if isinstance(err, RateLimited):
                        err.scope = p.rate_limit_scope(r.text)
                    raise err
                async for line in r.aiter_lines():
                    delta = _sse_delta(line)
                    if delta:
                        yield delta
        except httpx.TimeoutException as e:
            raise Transient(p.name, 0, f"timeout: {e}")
        except httpx.TransportError as e:
            raise Transient(p.name, 0, f"transport: {e}")

    async def _ado(self, cand: Candidate, req) -> ChatResponse:
        p = cand.provider
        client = self._ensure_client()
        body = req.payload(cand.model)
        t0 = time.monotonic()
        try:
            r = await client.post(p.url, headers=p.headers(cand.key.key), json=body)
        except httpx.TimeoutException as e:
            raise Transient(p.name, 0, f"timeout: {e}")
        except httpx.TransportError as e:
            raise Transient(p.name, 0, f"transport: {e}")
        dt = (time.monotonic() - t0) * 1000.0
        if r.status_code == 200:
            return p.parse_response(r.json(), dt)
        from .errors import RateLimited, classify

        err = classify(r.status_code, dict(r.headers), r.text, p.name)
        if isinstance(err, RateLimited):
            err.scope = p.rate_limit_scope(r.text)
        raise err

    async def aclose(self) -> None:
        self._save_state()
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncFreeLLM":
        self._ensure_client()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
