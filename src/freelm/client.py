"""The user-facing clients: ``FreeLLM`` (sync) and ``AsyncFreeLLM`` (async)."""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from concurrent import futures
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


def _repair_chunk(chunk: Dict[str, Any]) -> None:
    """Workers AI sometimes streams a numeric token as a JSON number
    (``"content": 6``); make it text before anyone joins the deltas."""
    for c in chunk.get("choices") or []:
        d = c.get("delta") if isinstance(c, dict) else None
        if isinstance(d, dict):
            v = d.get("content")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                d["content"] = str(v)


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


# A host that won't even accept the connection fails within this many seconds
# (Node's fetch does the same), however long the call's deadline is.
CONNECT_TIMEOUT = 10.0


def _timeout(seconds: Optional[float]) -> httpx.Timeout:
    if not seconds:
        return httpx.Timeout(None, connect=CONNECT_TIMEOUT)
    return httpx.Timeout(seconds, connect=min(CONNECT_TIMEOUT, seconds))


def _attempt_timeout(default: float, deadline: Optional[float]) -> httpx.Timeout:
    """One attempt may not outlive the call's overall deadline."""
    if deadline is None:
        return _timeout(default)
    return _timeout(max(0.1, deadline - time.monotonic()))


def _spawn(fn: Callable[..., Any], *args: Any) -> "futures.Future[Any]":
    """Run ``fn`` in a daemon thread and return its Future (an abandoned slow
    attempt must never hold up interpreter exit, so no thread pool)."""
    fut: "futures.Future[Any]" = futures.Future()

    def run() -> None:
        try:
            fut.set_result(fn(*args))
        except BaseException as e:  # delivered to whoever waits on the future
            fut.set_exception(e)

    threading.Thread(target=run, name="freelm-attempt", daemon=True).start()
    return fut


def _release(fut: Any, cleanup: Optional[Callable[[Any], Any]]) -> None:
    """Done-callback for an abandoned attempt: free what it produced (an open
    stream) and swallow its outcome."""
    try:
        if fut.cancelled() or fut.exception() is not None:
            return
        if cleanup is not None:
            out = cleanup(fut.result())
            if asyncio.iscoroutine(out):
                asyncio.ensure_future(out)
    except Exception:
        pass


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
        strategy: str = "smart",
        max_attempts: int = 12,
        timeout: float = 60.0,
        wait: bool = False,
        max_wait: float = 20.0,
        hedge: Union[bool, float] = True,
        on_event: Optional[Callable[[Event], Any]] = None,
        persist: Optional[bool] = None,
    ) -> None:
        providers = list(providers)
        if not providers:
            raise ConfigError("FreeLLM needs at least one provider")
        if strategy not in STRATEGIES:
            raise ConfigError(f"unknown strategy {strategy!r}; pick one of {STRATEGIES}")
        if not isinstance(hedge, (bool, int, float)) or hedge < 0:
            raise ConfigError("hedge must be True (adaptive), a delay in seconds, or False")
        self.providers = providers
        self.strategy = strategy
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.wait = wait
        self.max_wait = max_wait
        # True = adaptive hedging, a number = fixed delay (s), False/0 = sequential
        self.hedge: Union[bool, float] = hedge
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
        engine.apply_success(cand, latency_ms, time.monotonic())
        self._emit("success", cand=cand, latency_ms=latency_ms, attempt=len(attempts) + 1)
        self._save_state()

    def _exhausted(self, attempts: List) -> Exception:
        return engine.exhausted(attempts, self.providers, time.monotonic())

    # -- the race: one attempt at a time, plus a hedge when it's slow --------
    def _pick(self, req: Any, tried: set, attempts: List, running: int, now: float) -> Optional[Candidate]:
        """The next candidate to start (its rpm token reserved), or None."""
        while len(attempts) + running < self.max_attempts:
            cand = engine.select_candidate(
                self.providers, self.strategy, self._rr, req.model, tried, now, engine.rejected_by(attempts)
            )
            if cand is None:
                return None
            tried.add((cand.provider.name, cand.key.key, cand.model))
            if cand.key.reserve(now):
                return cand
        return None

    def _hedge_at(self, running: Dict[Any, Tuple[Candidate, float]], stream: bool) -> Optional[float]:
        """When the single running attempt gets a parallel hedge (monotonic), or None."""
        if len(running) != 1:
            return None
        ((cand, t0),) = running.values()
        d = engine.hedge_delay(cand, self.hedge, stream)
        return None if d is None else t0 + d

    def _settle(self, cand: Candidate, fut: Any, attempts: List) -> Tuple[bool, Any]:
        """Outcome of a finished attempt: ``(True, result)``, or ``(False, None)``
        after recording a failure that should fail over (raises otherwise)."""
        try:
            return True, fut.result()
        except ProviderError as exc:
            self._record_error(cand, exc, attempts)
            if engine.should_raise(exc, attempts):
                raise
            return False, None

    def _timed_out(self, running: Dict[Any, Tuple[Candidate, float]], attempts: List) -> None:
        for cand, _t0 in running.values():
            self._record_error(cand, Transient(cand.provider.name, 0, "timeout: no answer within the call's deadline"),
                               attempts)

    def _abandon(self, running: Dict[Any, Tuple[Candidate, float]], winner_t0: Optional[float],
                 cleanup: Optional[Callable[[Any], Any]]) -> None:
        """Leave attempts that lost the race (or outlived the call). One that
        started before the winner was slower than it: remember that. Whatever
        they still produce is released when it lands."""
        now = time.monotonic()
        for fut, (cand, t0) in list(running.items()):
            if winner_t0 is not None and t0 < winner_t0:
                engine.apply_slow(cand, (now - t0) * 1000.0, now)
            if isinstance(fut, asyncio.Future):
                fut.cancel()
            fut.add_done_callback(lambda f: _release(f, cleanup))
        running.clear()

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
        self._client = http_client or httpx.Client(timeout=_timeout(self.timeout), headers={"User-Agent": _DEFAULT_UA})
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
        # every provider's /models at once: the first call waits for the
        # slowest catalog, not the sum of them
        jobs = [(p, _spawn(discovery.discover_sync, p, self._client))
                for p in self.providers if getattr(p, "discover", False)]
        futures.wait([f for _, f in jobs])
        for p, f in jobs:
            if f.exception() is None and f.result():
                self._emit("discovery", provider=p.name)
            # (a failed discovery keeps the hardcoded fallback models)
        self._discovery_done = True

    def _race(self, req: Any, deadline: Optional[float], work: Callable[[Candidate], Any], *, stream: bool,
              cleanup: Optional[Callable[[Any], Any]] = None) -> Tuple[Candidate, Any, List]:
        """Run attempts until one succeeds and return ``(cand, result, attempts)``.

        One attempt at a time; when it is still running after the hedge delay
        (``hedge``), the next candidate starts in parallel and the first answer
        wins. ``work(cand)`` does the HTTP and returns a result or raises
        ``ProviderError``; with hedging on it runs in a worker thread, and all
        bookkeeping stays on this thread."""
        attempts: List = []
        tried: set = set()
        running: Dict[Any, Tuple[Candidate, float]] = {}
        blocked = False  # a hedge was due but no candidate was ready
        try:
            while True:
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    break
                hedge_at = None if blocked else self._hedge_at(running, stream)
                if not running or (hedge_at is not None and now >= hedge_at):
                    cand = self._pick(req, tried, attempts, len(running), now)
                    if cand is not None:
                        self._emit("hedge" if running else "attempt", cand=cand, attempt=len(attempts) + len(running) + 1)
                        if not self.hedge:  # sequential: no threads at all
                            fut: "futures.Future[Any]" = futures.Future()
                            try:
                                fut.set_result(work(cand))
                            except ProviderError as exc:
                                fut.set_exception(exc)
                            ok, result = self._settle(cand, fut, attempts)
                            if ok:
                                return cand, result, attempts
                            continue
                        running[_spawn(work, cand)] = (cand, time.monotonic())
                        continue
                    if not running:
                        w = self._wait_for(now, deadline, req.model, tried, attempts)
                        if w is None:
                            break
                        self._emit("wait", latency_ms=w * 1000.0, attempt=len(attempts))
                        time.sleep(w + 0.01)
                        fresh = engine.forget_recovered(self.providers, tried, time.monotonic())
                        tried.clear()
                        tried.update(fresh)
                        continue
                    blocked, hedge_at = True, None
                wake = [t for t in (deadline, hedge_at) if t is not None]
                timeout = max(0.0, min(wake) - now) if wake else None
                done, _ = futures.wait(list(running), timeout=timeout, return_when=futures.FIRST_COMPLETED)
                for f in done:
                    cand, t0 = running.pop(f)
                    blocked = False
                    ok, result = self._settle(cand, f, attempts)
                    if ok:
                        self._abandon(running, t0, cleanup)
                        return cand, result, attempts
            self._timed_out(running, attempts)
            raise self._exhausted(attempts)
        finally:
            self._abandon(running, None, cleanup)

    def chat(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> ChatResponse:
        _no_stream_kw(kw)
        self._ensure_discovered()
        req = build_request(messages, model, kw)
        deadline = time.monotonic() + self.timeout if self.timeout else None
        cand, resp, attempts = self._race(req, deadline, lambda c: self._do(c, req, deadline), stream=False)
        self._record_success(cand, resp.latency_ms, attempts)
        return resp

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
        cand, opened, attempts = self._race(
            req, deadline, lambda c: self._open(c, req, raw), stream=True, cleanup=lambda o: o[0].close()
        )
        gen, items, done, first_ms = opened
        try:
            yield from items
            if not done:
                for chunk in gen:
                    if raw:
                        yield chunk
                    else:
                        text = _chunk_text(chunk)
                        if text:
                            yield text
        except ProviderError as exc:
            self._record_error(cand, exc, attempts)
            raise  # output already reached the caller: no mid-stream failover
        finally:
            gen.close()
        self._record_success(cand, first_ms, attempts)

    def _open(self, cand: Candidate, req: Any, raw: bool) -> Tuple[Any, List[Any], bool, float]:
        """Start a stream and read up to its first emittable item. Returns
        ``(gen, items, done, first_ms)``: ``items`` go out first (raw mode: the
        held-back role-only preamble plus the first output chunk); ``done``
        means the stream ended without output (an empty completion)."""
        t0 = time.monotonic()
        gen = self._stream_do(cand, req)
        pending: List[Any] = []
        try:
            for chunk in gen:
                if raw:
                    if not _has_output(chunk):
                        pending.append(chunk)
                        continue
                    return gen, pending + [chunk], False, (time.monotonic() - t0) * 1000.0
                text = _chunk_text(chunk)
                if text:
                    return gen, [text], False, (time.monotonic() - t0) * 1000.0
        except BaseException:
            gen.close()
            raise
        return gen, pending, True, 0.0

    def _stream_do(self, cand: Candidate, req) -> Iterator[Dict[str, Any]]:
        p = cand.provider
        body = req.payload(cand.model)
        body["stream"] = True
        body = p.adapt_payload(body)
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
                    _repair_chunk(obj)
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
        body = p.adapt_payload(req.payload(cand.model))
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
            self._client = httpx.AsyncClient(timeout=_timeout(self.timeout), headers={"User-Agent": _DEFAULT_UA})
        return self._client

    async def _ensure_discovered(self) -> None:
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

    async def _race(self, req: Any, deadline: Optional[float], work: Callable[[Candidate], Any], *, stream: bool,
                    cleanup: Optional[Callable[[Any], Any]] = None) -> Tuple[Candidate, Any, List]:
        """Async twin of :meth:`FreeLLM._race`: attempts are tasks, and a
        hedge that loses is cancelled outright."""
        attempts: List = []
        tried: set = set()
        running: Dict[Any, Tuple[Candidate, float]] = {}
        blocked = False
        try:
            while True:
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    break
                hedge_at = None if blocked else self._hedge_at(running, stream)
                if not running or (hedge_at is not None and now >= hedge_at):
                    cand = self._pick(req, tried, attempts, len(running), now)
                    if cand is not None:
                        self._emit("hedge" if running else "attempt", cand=cand, attempt=len(attempts) + len(running) + 1)
                        if not self.hedge:  # sequential
                            fut: "asyncio.Future[Any]" = asyncio.get_running_loop().create_future()
                            try:
                                fut.set_result(await work(cand))
                            except ProviderError as exc:
                                fut.set_exception(exc)
                            ok, result = self._settle(cand, fut, attempts)
                            if ok:
                                return cand, result, attempts
                            continue
                        running[asyncio.ensure_future(work(cand))] = (cand, time.monotonic())
                        continue
                    if not running:
                        w = self._wait_for(now, deadline, req.model, tried, attempts)
                        if w is None:
                            break
                        self._emit("wait", latency_ms=w * 1000.0, attempt=len(attempts))
                        await asyncio.sleep(w + 0.01)
                        fresh = engine.forget_recovered(self.providers, tried, time.monotonic())
                        tried.clear()
                        tried.update(fresh)
                        continue
                    blocked, hedge_at = True, None
                wake = [t for t in (deadline, hedge_at) if t is not None]
                timeout = max(0.0, min(wake) - now) if wake else None
                done, _ = await asyncio.wait(list(running), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                for f in done:
                    cand, t0 = running.pop(f)
                    blocked = False
                    ok, result = self._settle(cand, f, attempts)
                    if ok:
                        self._abandon(running, t0, cleanup)
                        return cand, result, attempts
            self._timed_out(running, attempts)
            raise self._exhausted(attempts)
        finally:
            self._abandon(running, None, cleanup)

    async def chat(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> ChatResponse:
        _no_stream_kw(kw)
        await self._ensure_discovered()
        req = build_request(messages, model, kw)
        deadline = time.monotonic() + self.timeout if self.timeout else None
        cand, resp, attempts = await self._race(req, deadline, lambda c: self._ado(c, req, deadline), stream=False)
        self._record_success(cand, resp.latency_ms, attempts)
        return resp

    async def text(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> str:
        return (await self.chat(messages, model=model, **kw)).text

    def astream(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> AsyncIterator[str]:
        """Async content-delta stream. Fails over before the first token only."""
        return self._astream(messages, model, kw, raw=False)

    def astream_chunks(self, messages: Any, model: ModelArg = "auto", **kw: Any) -> AsyncIterator[Dict[str, Any]]:
        """Async raw ``chat.completion.chunk`` dicts (see :meth:`FreeLLM.stream_chunks`)."""
        return self._astream(messages, model, kw, raw=True)

    async def _astream(self, messages: Any, model: ModelArg, kw: Dict[str, Any], *, raw: bool) -> AsyncIterator[Any]:
        await self._ensure_discovered()
        req = build_request(messages, model, kw)
        deadline = time.monotonic() + self.timeout if self.timeout else None
        cand, opened, attempts = await self._race(
            req, deadline, lambda c: self._aopen(c, req, raw), stream=True, cleanup=lambda o: o[0].aclose()
        )
        gen, items, done, first_ms = opened
        try:
            for o in items:
                yield o
            if not done:
                async for chunk in gen:
                    if raw:
                        yield chunk
                    else:
                        text = _chunk_text(chunk)
                        if text:
                            yield text
        except ProviderError as exc:
            self._record_error(cand, exc, attempts)
            raise  # output already reached the caller: no mid-stream failover
        finally:
            await gen.aclose()
        self._record_success(cand, first_ms, attempts)

    async def _aopen(self, cand: Candidate, req: Any, raw: bool) -> Tuple[Any, List[Any], bool, float]:
        """Async twin of :meth:`FreeLLM._open`."""
        t0 = time.monotonic()
        gen = self._astream_do(cand, req)
        pending: List[Any] = []
        try:
            async for chunk in gen:
                if raw:
                    if not _has_output(chunk):
                        pending.append(chunk)
                        continue
                    return gen, pending + [chunk], False, (time.monotonic() - t0) * 1000.0
                text = _chunk_text(chunk)
                if text:
                    return gen, [text], False, (time.monotonic() - t0) * 1000.0
        except BaseException:
            await gen.aclose()
            raise
        return gen, pending, True, 0.0

    async def _astream_do(self, cand: Candidate, req) -> AsyncIterator[Dict[str, Any]]:
        p = cand.provider
        client = self._ensure_client()
        body = req.payload(cand.model)
        body["stream"] = True
        body = p.adapt_payload(body)
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
                    _repair_chunk(obj)
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
        body = p.adapt_payload(req.payload(cand.model))
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
