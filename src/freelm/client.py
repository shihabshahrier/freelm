"""The user-facing clients: ``FreeLLM`` (sync) and ``AsyncFreeLLM`` (async)."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import httpx

from . import _engine as engine
from . import discovery
from ._state import StateStore
from ._types import ChatResponse, Event, build_request
from ._version import __version__
from .errors import ConfigError, ProviderError, RateLimited, Transient, classify
from .providers.base import Provider
from .strategy import STRATEGIES, Candidate

_DEFAULT_UA = f"freelm/{__version__}"

ModelArg = Union[str, Sequence[str]]


# -- SSE / response helpers ---------------------------------------------------


def _sse_json(line: str) -> Optional[Dict[str, Any]]:
    """Parse one OpenAI-style SSE ``data:`` line into a dict, or None."""
    if not line or not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        obj = json.loads(data)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


_NEWLINE = re.compile(r"\r\n|\r|\n")


def _split_lines(buf: str, final: bool) -> Tuple[List[str], str]:
    """Complete lines in ``buf`` (split on CR, LF or CRLF only — never on
    U+2028/U+0085, which JSON strings may contain), plus the remainder."""
    out: List[str] = []
    pos = 0
    for m in _NEWLINE.finditer(buf):
        if m.group() == "\r" and m.end() == len(buf) and not final:
            break  # may be the first half of a CRLF split across chunks
        out.append(buf[pos:m.start()])
        pos = m.end()
    rest = buf[pos:]
    if final and rest:
        out.append(rest)
        rest = ""
    return out, rest


def _iter_lines(chunks: Iterator[str]) -> Iterator[str]:
    buf = ""
    for text in chunks:
        lines, buf = _split_lines(buf + text, final=False)
        yield from lines
    lines, _ = _split_lines(buf, final=True)
    yield from lines


async def _aiter_lines(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    buf = ""
    async for text in chunks:
        lines, buf = _split_lines(buf + text, final=False)
        for line in lines:
            yield line
    lines, _ = _split_lines(buf, final=True)
    for line in lines:
        yield line


_NOT_JSON = object()


class _SSE:
    """Incremental decoder for OpenAI-style SSE: one JSON object per ``data:``
    event. Handles comments/``event:`` lines, ``[DONE]``, multi-line ``data:``
    events, and servers that skip the blank separator line."""

    def __init__(self) -> None:
        self.buf: List[str] = []
        self.done = False

    def feed(self, line: str) -> Optional[Dict[str, Any]]:
        line = line.rstrip("\r")
        if not line:  # blank line = end of event
            self.buf = []
            return None
        if not line.startswith("data:"):
            return None  # ": keep-alive" comments, event:, id:, retry:
        data = line[5:]
        if data.startswith(" "):
            data = data[1:]
        if not self.buf and data.strip() == "[DONE]":
            self.done = True
            return None
        self.buf.append(data)
        obj = _loads("\n".join(self.buf))
        if obj is _NOT_JSON and len(self.buf) > 1:
            obj = _loads(data)  # a stale fragment must not swallow a whole valid line
        if obj is _NOT_JSON:
            return None  # an incomplete multi-line event: wait for more
        self.buf = []  # a complete event (even `null`) ends the buffer
        return obj if isinstance(obj, dict) else None


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return _NOT_JSON


def _chunk_text(chunk: Dict[str, Any]) -> Optional[str]:
    """The text delta of a ``chat.completion.chunk`` dict (content-part lists
    flattened), or None."""
    try:
        content = chunk["choices"][0]["delta"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    if isinstance(content, list):
        content = "".join(p.get("text") or "" for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content if isinstance(content, str) and content else None


def _sse_delta(line: str) -> Optional[str]:
    """Extract the content delta from one OpenAI-style SSE line, or None."""
    obj = _sse_json(line)
    return _chunk_text(obj) if obj is not None else None


def _has_output(chunk: Dict[str, Any]) -> bool:
    """Does this chunk carry anything a consumer would act on (text, tool
    calls, reasoning, a finish reason)? Role-only preambles don't."""
    choices = chunk.get("choices")
    for c in choices if isinstance(choices, list) else []:
        if not isinstance(c, dict):
            continue
        if c.get("finish_reason"):
            return True
        d = c.get("delta") or {}
        if isinstance(d, dict) and (
            d.get("content") or d.get("tool_calls") or d.get("reasoning") or d.get("reasoning_content")
        ):
            return True
    return False


def _classify(p: Provider, status: int, headers: Optional[Dict[str, str]], body: str) -> ProviderError:
    """``classify`` plus the provider's say on whether the error is key- or model-scoped."""
    err = classify(status, headers, body, p.name)
    if isinstance(err, RateLimited):
        err.scope = p.rate_limit_scope(body)
    elif isinstance(err, Transient):
        err.scope = p.transient_scope(body)
    return err


def _error_frame(p: Provider, err: Any) -> ProviderError:
    """An ``{"error": ...}`` object delivered with HTTP 200 (whole body or a
    mid-stream SSE frame). Uses its numeric ``code`` when present; otherwise
    it's treated as an upstream failure (502)."""
    code = err.get("code") if isinstance(err, dict) else None
    status = code if isinstance(code, int) and 400 <= code < 600 else 502
    body = err if isinstance(err, str) else json.dumps({"error": err})
    return _classify(p, status, None, body)


def _attempt_timeout(default: float, deadline: Optional[float]) -> Optional[float]:
    """One attempt may not outlive the call's overall deadline."""
    if deadline is None:
        return default or None
    return max(0.1, deadline - time.monotonic())


def _no_stream_kw(kw: Dict[str, Any]) -> None:
    if kw.pop("stream", False):
        raise ConfigError("chat() returns a whole response; use stream() / stream_chunks() to stream")


def _parse_ok(p: Provider, r: httpx.Response, latency_ms: float) -> ChatResponse:
    try:
        data = r.json()
    except ValueError:
        raise Transient(p.name, r.status_code, "invalid JSON in response body")
    if not isinstance(data, dict):
        raise Transient(p.name, r.status_code, "unexpected response shape")
    if data.get("error"):
        raise _error_frame(p, data["error"])
    return p.parse_response(data, latency_ms)


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

    # -- shared loop pieces -----------------------------------------------
    def _wait_for(self, now: float, deadline: Optional[float], alias: Any, tried: set, attempts: List) -> Optional[float]:
        """Seconds to sleep when no candidate is ready, or None to give up: the
        soonest an untried (or recovering) candidate frees up, counting cooling
        keys *and* benched models. None when waiting can't help."""
        if not self.wait:
            return None
        w = engine.soonest_wait(self.providers, now, alias, tried, engine.rejected_by(attempts))
        if w is None or w <= 0 or w > self.max_wait:
            return None
        if deadline is not None and now + w >= deadline:
            return None
        return w

    def _record_error(self, cand: Candidate, exc: ProviderError, attempts: List) -> None:
        engine.apply_error(cand, exc, time.monotonic())
        attempts.append((cand, exc))
        self._emit("error", cand=cand, status=exc.status, error=str(exc), attempt=len(attempts))
        self._save_state()

    def _record_success(self, cand: Candidate, latency_ms: float, attempts: List) -> None:
        engine.apply_success(cand, latency_ms)
        self._emit("success", cand=cand, latency_ms=latency_ms, attempt=len(attempts) + 1)
        self._save_state()

    def _exhausted(self, attempts: List) -> Exception:
        return engine.exhausted(attempts, self.providers, time.monotonic())

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

    def reset_keys(self) -> None:
        """Give every key and model a clean slate: re-enable disabled keys, clear
        cooldowns, breakers and benched models (e.g. after fixing a key or
        topping up credits)."""
        from ._breaker import CircuitBreaker

        for p in self.providers:
            p._model_until.clear()
            for k in p.keys:
                k.model_until.clear()
                k.disabled = False
                k.disabled_since_wall = 0.0
                k.cooldown_until = 0.0
                k.breaker = CircuitBreaker()
                k.last_error = None
        self._save_state()

    def refresh_models(self) -> None:
        """Force a live re-discovery on the next call (bypasses the in-memory
        guard *and* the disk cache)."""
        from . import _cache

        self._discovery_done = False
        for p in self.providers:
            p._discovered = False
            if getattr(p, "discover", False):
                _cache.clear(p.name)


class FreeLLM(_BaseClient):
    """Synchronous always-up chat client."""

    def __init__(self, providers: Sequence[Provider], *, http_client: Optional[httpx.Client] = None, **kw: Any) -> None:
        super().__init__(providers, **kw)
        self._client = http_client or httpx.Client(timeout=self.timeout, headers={"User-Agent": _DEFAULT_UA})
        self._owns_client = http_client is None

    @classmethod
    def from_env(cls, *, keyless: Any = None, **kw: Any) -> "FreeLLM":
        """Providers from environment keys; ``keyless=True`` / ``"auto"`` adds the
        no-signup endpoints (always / only when no keys are set)."""
        from .config import providers_from_env

        return cls(providers_from_env(keyless), **kw)

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

    def chat(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> ChatResponse:
        _no_stream_kw(kw)
        self._ensure_discovered()
        req = build_request(messages, model, kw)
        deadline = time.monotonic() + self.timeout if self.timeout else None
        attempts: List = []
        tried: set = set()

        while len(attempts) < self.max_attempts:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            cand = engine.select_candidate(
                self.providers, self.strategy, self._rr, req.model, tried, now, engine.rejected_by(attempts)
            )
            if cand is None:
                w = self._wait_for(now, deadline, req.model, tried, attempts)
                if w is None:
                    break
                self._emit("wait", latency_ms=w * 1000.0, attempt=len(attempts))
                time.sleep(w + 0.01)
                tried = engine.forget_recovered(self.providers, tried, time.monotonic())
                continue

            tried.add((cand.provider.name, cand.key.key, cand.model))
            if not cand.key.reserve(now):
                continue  # lost an rpm token to a concurrent caller; pick another

            self._emit("attempt", cand=cand, attempt=len(attempts) + 1)
            try:
                resp = self._do(cand, req, deadline)
            except ProviderError as exc:
                self._record_error(cand, exc, attempts)
                if engine.should_raise(exc, attempts):
                    raise
                continue
            self._record_success(cand, resp.latency_ms, attempts)
            return resp

        raise self._exhausted(attempts)

    def text(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> str:
        return self.chat(messages, model=model, **kw).text

    def stream(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> Iterator[str]:
        """Yield content deltas as they arrive. Fails over between providers
        *before* the first token; once tokens start flowing it stays on that
        provider (no mid-stream failover)."""
        return self._stream(messages, model, kw, raw=False)

    def stream_chunks(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> Iterator[Dict[str, Any]]:
        """Like :meth:`stream`, but yield the raw OpenAI ``chat.completion.chunk``
        dicts — tool-call deltas, finish reasons and usage included. Chunks
        that carry no output yet (a role-only preamble) are held back until the
        first real one, so failover stays invisible to the consumer."""
        return self._stream(messages, model, kw, raw=True)

    def _stream(self, messages: Any, model: ModelArg, kw: Dict[str, Any], *, raw: bool) -> Iterator[Any]:
        self._ensure_discovered()
        req = build_request(messages, model, kw)
        deadline = time.monotonic() + self.timeout if self.timeout else None
        attempts: List = []
        tried: set = set()

        while len(attempts) < self.max_attempts:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            cand = engine.select_candidate(
                self.providers, self.strategy, self._rr, req.model, tried, now, engine.rejected_by(attempts)
            )
            if cand is None:
                w = self._wait_for(now, deadline, req.model, tried, attempts)
                if w is None:
                    break
                self._emit("wait", latency_ms=w * 1000.0, attempt=len(attempts))
                time.sleep(w + 0.01)
                tried = engine.forget_recovered(self.providers, tried, time.monotonic())
                continue

            tried.add((cand.provider.name, cand.key.key, cand.model))
            if not cand.key.reserve(now):
                continue

            produced = False
            first_ms = 0.0  # time-to-first-token; feeds the latency EWMA
            pending: List[Dict[str, Any]] = []  # raw mode: chunks held until one carries output
            t0 = time.monotonic()
            self._emit("attempt", cand=cand, attempt=len(attempts) + 1)
            try:
                for chunk in self._stream_do(cand, req):
                    if raw:
                        if not produced and not _has_output(chunk):
                            pending.append(chunk)
                            continue
                        out: List[Any] = pending + [chunk]
                        pending = []
                    else:
                        text = _chunk_text(chunk)
                        if not text:
                            continue
                        out = [text]
                    if not produced:
                        first_ms = (time.monotonic() - t0) * 1000.0
                        produced = True
                    yield from out
            except ProviderError as exc:
                self._record_error(cand, exc, attempts)
                if produced or engine.should_raise(exc, attempts):
                    raise
                continue
            yield from pending  # a raw stream that never carried output (empty completion)
            self._record_success(cand, first_ms, attempts)
            return

        raise self._exhausted(attempts)

    def _stream_do(self, cand: Candidate, req) -> Iterator[Dict[str, Any]]:
        p = cand.provider
        body = req.payload(cand.model)
        body["stream"] = True
        try:
            with self._client.stream("POST", p.url, headers=p.headers(cand.key.key), json=body) as r:
                if r.status_code != 200:
                    r.read()
                    raise _classify(p, r.status_code, dict(r.headers), r.text)
                sse = _SSE()
                output = False
                for line in _iter_lines(r.iter_text()):
                    obj = sse.feed(line)
                    if sse.done:
                        break
                    if obj is None:
                        continue
                    if obj.get("error"):
                        raise _error_frame(p, obj["error"])
                    output = output or _has_output(obj)
                    yield obj
                if not sse.done and not output:
                    # an empty or cut-off 200 before any output: fail over
                    raise Transient(p.name, 200, "stream ended before any output")
        except httpx.TimeoutException as e:
            raise Transient(p.name, 0, f"timeout: {e}")
        except httpx.TransportError as e:
            raise Transient(p.name, 0, f"transport: {e}")

    def _do(self, cand: Candidate, req, deadline: Optional[float] = None) -> ChatResponse:
        p = cand.provider
        body = req.payload(cand.model)
        t0 = time.monotonic()
        try:
            r = self._client.post(
                p.url, headers=p.headers(cand.key.key), json=body, timeout=_attempt_timeout(self.timeout, deadline)
            )
        except httpx.TimeoutException as e:
            raise Transient(p.name, 0, f"timeout: {e}")
        except httpx.TransportError as e:
            raise Transient(p.name, 0, f"transport: {e}")
        dt = (time.monotonic() - t0) * 1000.0
        if r.status_code == 200:
            return _parse_ok(p, r, dt)
        raise _classify(p, r.status_code, dict(r.headers), r.text)

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
        self._discovery_lock: Any = None  # asyncio.Lock, created inside the running loop

    @classmethod
    def from_env(cls, *, keyless: Any = None, **kw: Any) -> "AsyncFreeLLM":
        from .config import providers_from_env

        return cls(providers_from_env(keyless), **kw)

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, headers={"User-Agent": _DEFAULT_UA})
        return self._client

    async def _ensure_discovered(self) -> None:
        import asyncio

        if self._discovery_done:
            return
        if self._discovery_lock is None:
            self._discovery_lock = asyncio.Lock()
        async with self._discovery_lock:  # N concurrent first calls -> one discovery
            if self._discovery_done:
                return
            client = self._ensure_client()

            async def one(p: Provider) -> None:
                try:
                    if await discovery.discover_async(p, client):
                        self._emit("discovery", provider=p.name)
                except Exception:
                    pass  # keep hardcoded fallback models

            await asyncio.gather(*(one(p) for p in self.providers if getattr(p, "discover", False)))
            self._discovery_done = True

    async def chat(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> ChatResponse:
        import asyncio

        _no_stream_kw(kw)
        await self._ensure_discovered()
        req = build_request(messages, model, kw)
        deadline = time.monotonic() + self.timeout if self.timeout else None
        attempts: List = []
        tried: set = set()

        while len(attempts) < self.max_attempts:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            cand = engine.select_candidate(
                self.providers, self.strategy, self._rr, req.model, tried, now, engine.rejected_by(attempts)
            )
            if cand is None:
                w = self._wait_for(now, deadline, req.model, tried, attempts)
                if w is None:
                    break
                self._emit("wait", latency_ms=w * 1000.0, attempt=len(attempts))
                await asyncio.sleep(w + 0.01)
                tried = engine.forget_recovered(self.providers, tried, time.monotonic())
                continue

            tried.add((cand.provider.name, cand.key.key, cand.model))
            if not cand.key.reserve(now):
                continue

            self._emit("attempt", cand=cand, attempt=len(attempts) + 1)
            try:
                resp = await self._ado(cand, req, deadline)
            except ProviderError as exc:
                self._record_error(cand, exc, attempts)
                if engine.should_raise(exc, attempts):
                    raise
                continue
            self._record_success(cand, resp.latency_ms, attempts)
            return resp

        raise self._exhausted(attempts)

    async def text(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> str:
        return (await self.chat(messages, model=model, **kw)).text

    def astream(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> AsyncIterator[str]:
        """Async content-delta stream. Fails over before the first token only."""
        return self._astream(messages, model, kw, raw=False)

    def astream_chunks(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> AsyncIterator[Dict[str, Any]]:
        """Async raw ``chat.completion.chunk`` dicts (see :meth:`FreeLLM.stream_chunks`)."""
        return self._astream(messages, model, kw, raw=True)

    async def _astream(self, messages: Any, model: ModelArg, kw: Dict[str, Any], *, raw: bool) -> AsyncIterator[Any]:
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
            cand = engine.select_candidate(
                self.providers, self.strategy, self._rr, req.model, tried, now, engine.rejected_by(attempts)
            )
            if cand is None:
                w = self._wait_for(now, deadline, req.model, tried, attempts)
                if w is None:
                    break
                self._emit("wait", latency_ms=w * 1000.0, attempt=len(attempts))
                await asyncio.sleep(w + 0.01)
                tried = engine.forget_recovered(self.providers, tried, time.monotonic())
                continue

            tried.add((cand.provider.name, cand.key.key, cand.model))
            if not cand.key.reserve(now):
                continue

            produced = False
            first_ms = 0.0  # time-to-first-token; feeds the latency EWMA
            pending: List[Dict[str, Any]] = []
            t0 = time.monotonic()
            self._emit("attempt", cand=cand, attempt=len(attempts) + 1)
            try:
                async for chunk in self._astream_do(cand, req):
                    if raw:
                        if not produced and not _has_output(chunk):
                            pending.append(chunk)
                            continue
                        out: List[Any] = pending + [chunk]
                        pending = []
                    else:
                        text = _chunk_text(chunk)
                        if not text:
                            continue
                        out = [text]
                    if not produced:
                        first_ms = (time.monotonic() - t0) * 1000.0
                        produced = True
                    for o in out:
                        yield o
            except ProviderError as exc:
                self._record_error(cand, exc, attempts)
                if produced or engine.should_raise(exc, attempts):
                    raise
                continue
            for o in pending:
                yield o
            self._record_success(cand, first_ms, attempts)
            return

        raise self._exhausted(attempts)

    async def _astream_do(self, cand: Candidate, req) -> AsyncIterator[Dict[str, Any]]:
        p = cand.provider
        client = self._ensure_client()
        body = req.payload(cand.model)
        body["stream"] = True
        try:
            async with client.stream("POST", p.url, headers=p.headers(cand.key.key), json=body) as r:
                if r.status_code != 200:
                    await r.aread()
                    raise _classify(p, r.status_code, dict(r.headers), r.text)
                sse = _SSE()
                output = False
                async for line in _aiter_lines(r.aiter_text()):
                    obj = sse.feed(line)
                    if sse.done:
                        break
                    if obj is None:
                        continue
                    if obj.get("error"):
                        raise _error_frame(p, obj["error"])
                    output = output or _has_output(obj)
                    yield obj
                if not sse.done and not output:
                    raise Transient(p.name, 200, "stream ended before any output")
        except httpx.TimeoutException as e:
            raise Transient(p.name, 0, f"timeout: {e}")
        except httpx.TransportError as e:
            raise Transient(p.name, 0, f"transport: {e}")

    async def _ado(self, cand: Candidate, req, deadline: Optional[float] = None) -> ChatResponse:
        p = cand.provider
        client = self._ensure_client()
        body = req.payload(cand.model)
        t0 = time.monotonic()
        try:
            r = await client.post(
                p.url, headers=p.headers(cand.key.key), json=body, timeout=_attempt_timeout(self.timeout, deadline)
            )
        except httpx.TimeoutException as e:
            raise Transient(p.name, 0, f"timeout: {e}")
        except httpx.TransportError as e:
            raise Transient(p.name, 0, f"transport: {e}")
        dt = (time.monotonic() - t0) * 1000.0
        if r.status_code == 200:
            return _parse_ok(p, r, dt)
        raise _classify(p, r.status_code, dict(r.headers), r.text)

    async def aclose(self) -> None:
        self._save_state()
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncFreeLLM":
        self._ensure_client()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
