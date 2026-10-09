"""``freelm serve`` — the router as a local OpenAI-compatible HTTP endpoint.

Point any OpenAI-compatible tool at it (Cursor, Cline, Continue, Open WebUI,
n8n, LangChain's ``ChatOpenAI``, the OpenAI SDKs, ...)::

    freelm serve                       # http://127.0.0.1:4000/v1
    OPENAI_BASE_URL=http://127.0.0.1:4000/v1 OPENAI_API_KEY=freelm your-tool

Endpoints: ``POST /v1/chat/completions`` (JSON or SSE with ``stream: true``),
``GET /v1/models`` (virtual aliases + discovered models), ``GET /health``.

stdlib only: ``ThreadingHTTPServer`` parses HTTP; every request is handed to
one ``AsyncFreeLLM`` living on a private event-loop thread, so key state stays
single-loop-safe however many requests run at once.

Browser safety: POSTs must be ``application/json`` (a web page can't send that
cross-origin without a CORS preflight, which is refused unless ``--cors``), and
while bound to loopback the ``Host`` header must be loopback too (DNS rebinding).
"""
from __future__ import annotations

import asyncio
import hmac
import json
import queue
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ._version import __version__
from .client import AsyncFreeLLM
from .errors import BadRequest, ConfigError, FreeLLMError, NoProvidersAvailable, ProviderError
from .registry import _VIRTUAL, is_virtual
from .types_compat import _ChunkStamper, wrap_completion

MAX_BODY = 20 * 1024 * 1024  # generous for base64 images, bounded against abuse
_CHAT_PATHS = ("/v1/chat/completions", "/chat/completions")
_MODEL_PATHS = ("/v1/models", "/models")
_HEALTH_PATHS = ("/health", "/v1/health", "/healthz")
_LOOPBACK = ("127.0.0.1", "localhost", "::1")
_DEFAULT_CORS_HEADERS = "Authorization, Content-Type, X-Api-Key"
_DONE = object()


class _BodyError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class _Loop:
    """An asyncio loop on a daemon thread; other threads submit coroutines to it."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="freelm-serve-loop", daemon=True)
        self.thread.start()

    def run(self, coro: Any, timeout: Optional[float] = None) -> Any:
        if not self.loop.is_running():
            coro.close()
            raise RuntimeError("freelm serve is shutting down")
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stream(self, agen_factory: Callable[[], Any]) -> Tuple[queue.SimpleQueue, Any]:
        """Pump an async generator into a thread-safe queue. Returns the queue and
        the future (cancel it to stop the upstream stream early)."""
        q: queue.SimpleQueue = queue.SimpleQueue()

        async def pump() -> None:
            try:
                async for item in agen_factory():
                    q.put(item)
            except BaseException as e:  # noqa: BLE001 - handed to the HTTP thread
                q.put(e)
                if isinstance(e, asyncio.CancelledError):
                    raise
            finally:
                q.put(_DONE)

        return q, asyncio.run_coroutine_threadsafe(pump(), self.loop)

    def close(self) -> None:
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


def _error_body(message: str, type_: str, code: Optional[str] = None) -> Dict[str, Any]:
    return {"error": {"message": message, "type": type_, "param": None, "code": code}}


def _status_for(e: BaseException) -> Tuple[int, Dict[str, Any]]:
    """Map a freelm error onto an OpenAI-style HTTP error."""
    if isinstance(e, ConfigError):
        return 400, _error_body(str(e), "invalid_request_error", "config_error")
    if isinstance(e, BadRequest):
        status = e.status if 400 <= e.status < 500 else 400
        return status, _error_body(str(e), "invalid_request_error", "rejected_by_providers")
    if isinstance(e, NoProvidersAvailable):
        return 503, _error_body(str(e), "service_unavailable", "no_providers_available")
    if isinstance(e, ProviderError):
        return 502, _error_body(str(e), "upstream_error", "provider_error")
    if isinstance(e, FreeLLMError):
        return 500, _error_body(str(e), "server_error")
    return 500, _error_body(f"internal error: {type(e).__name__}", "server_error")


def _host_only(host_header: str) -> str:
    h = host_header.strip().lower()
    if h.startswith("["):  # [::1]:4000
        return h[1:h.find("]")] if "]" in h else h
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


def url_for(host: str, port: int) -> str:
    """``http://host:port/v1`` with IPv6 hosts bracketed."""
    h = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{h}:{port}/v1"


class FreeLLMServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128  # the stdlib default of 5 drops bursts of parallel tool requests

    def __init__(
        self,
        address: Tuple[str, int],
        llm: AsyncFreeLLM,
        *,
        api_key: Optional[str] = None,
        cors: bool = False,
        fallback_model: Optional[str] = "auto",
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        # everything server_close() touches must exist before bind: a failed
        # bind calls server_close() from inside TCPServer.__init__
        self._closed = False
        self.llm = llm
        self.api_key = api_key
        self.cors = cors
        self.fallback_model = fallback_model
        self.log = log
        self.bridge = _Loop()
        self._warned: set = set()
        self.loopback_only = address[0] in _LOOPBACK
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, _Handler)

    def model_chain(self, requested: Any) -> Any:
        """A concrete id no provider offers (e.g. a tool's default "gpt-4o")
        falls back to ``fallback_model`` instead of failing every provider."""
        if not isinstance(requested, str) or not requested.strip():
            return self.fallback_model or "auto"
        if is_virtual(requested) or not self.fallback_model:
            return requested
        if any(p.knows_model(requested) for p in self.llm.providers):
            return requested
        if requested not in self._warned and self.log:
            self._warned.add(requested)
            self.log(f"note: no free provider lists model {requested!r}; trying it, then {self.fallback_model!r}")
        return [requested, self.fallback_model]

    def server_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.bridge.run(self.llm.aclose(), timeout=5)
        except Exception:
            pass
        self.bridge.close()
        super().server_close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: FreeLLMServer
    server_version = f"freelm/{__version__}"

    # -- plumbing ---------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # quiet default access log
        pass

    def _cors_headers(self) -> None:
        if self.server.cors:
            self.send_header("Access-Control-Allow-Origin", "*")
            # reflect what the browser asks for (openai-node sends x-stainless-* headers)
            asked = self.headers.get("Access-Control-Request-Headers")
            self.send_header("Access-Control-Allow-Headers", asked or _DEFAULT_CORS_HEADERS)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_json(self, status: int, obj: Any, headers: Optional[Dict[str, str]] = None, *, close: bool = False) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        if close:
            # the request body (if any) is left unread: never reuse this connection
            self.send_header("Connection", "close")
            self.close_connection = True
        self._cors_headers()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _guard(self, *, body_pending: bool) -> bool:
        """Host (DNS rebinding) and API-key checks. False = already answered."""
        if self.server.loopback_only and _host_only(self.headers.get("Host", "")) not in _LOOPBACK:
            self._send_json(403, _error_body("this freelm server only answers requests addressed to localhost",
                                             "invalid_request_error", "forbidden_host"), close=body_pending)
            return False
        key = self.server.api_key
        if not key:
            return True
        auth = self.headers.get("Authorization", "")
        given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("X-Api-Key", "")
        if given and hmac.compare_digest(given.encode(), key.encode()):
            return True
        self._send_json(401, _error_body("missing or invalid API key for this freelm server", "invalid_request_error",
                                         "invalid_api_key"), close=body_pending)
        return False

    def _path(self) -> str:
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def _read_body(self) -> bytes:
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            out = bytearray()
            while True:
                line = self.rfile.readline(1024)
                try:
                    size = int(line.split(b";", 1)[0].strip(), 16)
                except ValueError:
                    raise _BodyError(400, "malformed chunked request body")
                if size == 0:
                    while self.rfile.readline(1024) not in (b"\r\n", b"\n", b""):
                        pass  # trailers
                    return bytes(out)
                if len(out) + size > MAX_BODY:
                    raise _BodyError(413, f"request body larger than {MAX_BODY // (1024 * 1024)} MB")
                out += self.rfile.read(size)
                self.rfile.readline(1024)  # CRLF after each chunk
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise _BodyError(400, "invalid Content-Length")
        if length <= 0:
            raise _BodyError(400, "a JSON request body is required")
        if length > MAX_BODY:
            raise _BodyError(413, f"request body larger than {MAX_BODY // (1024 * 1024)} MB")
        return self.rfile.read(length)

    # -- verbs --------------------------------------------------------------
    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server API
        self.send_response(204)
        self._cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self._path()
        if path in _HEALTH_PATHS:
            llm = self.server.llm

            async def snapshot() -> List[Dict[str, Any]]:  # key state lives on the loop thread
                return llm.health()

            self._send_json(200, {"status": "ok", "version": __version__, "keys": self.server.bridge.run(snapshot())})
            return
        if not self._guard(body_pending=False):
            return
        if path in _MODEL_PATHS:
            self._send_json(200, self._models())
        elif path in ("/", "/v1"):
            self._send_json(200, {"name": "freelm", "version": __version__,
                                  "endpoints": ["/v1/chat/completions", "/v1/models", "/health"]})
        else:
            self._send_json(404, _error_body(f"no route for GET {path}", "invalid_request_error", "not_found"))

    do_HEAD = do_GET  # noqa: N815 - same routes, _send_json skips the body

    def _no_route(self) -> None:
        self._send_json(404, _error_body(f"no route for {self.command} {self._path()}", "invalid_request_error",
                                         "not_found"), close=True)

    do_PUT = do_DELETE = do_PATCH = _no_route  # noqa: N815

    def do_POST(self) -> None:  # noqa: N802
        path = self._path()
        if not self._guard(body_pending=True):
            return
        if path not in _CHAT_PATHS:
            self._no_route()
            return
        if "application/json" not in (self.headers.get("Content-Type") or "").lower():
            # also what keeps arbitrary web pages out: a cross-origin JSON POST
            # needs a CORS preflight, which is refused unless --cors
            self._send_json(415, _error_body("Content-Type must be application/json", "invalid_request_error"),
                            close=True)
            return
        try:
            body = json.loads(self._read_body())
            if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
                raise ValueError("body must be an object with a 'messages' list")
        except _BodyError as e:
            self._send_json(e.status, _error_body(str(e), "invalid_request_error"), close=True)
            return
        except ValueError as e:
            self._send_json(400, _error_body(f"invalid JSON body: {e}", "invalid_request_error"))
            return
        self._chat(body)

    # -- handlers -------------------------------------------------------------
    def _models(self) -> Dict[str, Any]:
        llm = self.server.llm
        try:
            self.server.bridge.run(llm._ensure_discovered())
        except Exception:
            pass
        now = int(time.time())
        data: List[Dict[str, Any]] = [{"id": a, "object": "model", "created": now, "owned_by": "freelm"} for a in sorted(_VIRTUAL)]
        seen = {d["id"] for d in data}
        for p in llm.providers:
            for m in p.models:
                if m.id not in seen:
                    seen.add(m.id)
                    data.append({"id": m.id, "object": "model", "created": now, "owned_by": p.name})
        return {"object": "list", "data": data}

    def _chat(self, body: Dict[str, Any]) -> None:
        t0 = time.monotonic()
        stream = bool(body.pop("stream", False))
        body.pop("stream_options", None)
        messages = body.pop("messages")
        model = self.server.model_chain(body.pop("model", None))
        kw = {k: v for k, v in body.items() if v is not None}
        llm = self.server.llm
        if not stream:
            try:
                resp = self.server.bridge.run(llm.chat(messages, model=model, **kw))
            except Exception as e:  # noqa: BLE001 - mapped to an HTTP error
                status, err = _status_for(e)
                self._send_json(status, err)
                self._log(status, None, t0, e)
                return
            out = wrap_completion(resp).model_dump()
            self._send_json(200, out, {"X-FreeLLM-Provider": resp.provider or "", "X-FreeLLM-Model": resp.model or ""})
            self._log(200, f"{resp.provider}/{resp.model}", t0)
            return

        q, fut = self.server.bridge.stream(lambda: llm.astream_chunks(messages, model=model, **kw))
        first = q.get()
        if isinstance(first, BaseException):
            status, err = _status_for(first)
            self._send_json(status, err)
            self._log(status, None, t0, first)
            return
        chunked = self.request_version == "HTTP/1.1"
        stamp = _ChunkStamper()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:  # HTTP/1.0 has no chunked encoding: stream raw, end by closing
            self.send_header("Connection", "close")
            self.close_connection = True
        self._cors_headers()
        self.end_headers()

        def write(data: bytes) -> None:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n" if chunked else data)
            self.wfile.flush()

        served = None
        try:
            item = first
            while item is not _DONE:
                if isinstance(item, BaseException):
                    _, err = _status_for(item)
                    write(b"data: " + json.dumps(err).encode() + b"\n\n")
                    break
                served = served or item.get("model")
                write(b"data: " + json.dumps(stamp(item).model_dump()).encode() + b"\n\n")
                item = q.get()
            write(b"data: [DONE]\n\n")
            if chunked:
                self.wfile.write(b"0\r\n\r\n")  # terminating chunk
                self.wfile.flush()
            self._log(200, f"stream/{served}", t0)
        except OSError:
            self.close_connection = True  # the client went away (any OS's flavour of it)
        finally:
            if not fut.done():
                fut.cancel()  # stop pulling tokens upstream

    def _log(self, status: int, served: Optional[str], t0: float, err: Optional[BaseException] = None) -> None:
        if self.server.log:
            ms = (time.monotonic() - t0) * 1000.0
            what = served or (type(err).__name__ if err else "-")
            self.server.log(f"{self.command} {self._path()} {status} {what} {ms:.0f}ms")


def make_server(
    llm: AsyncFreeLLM,
    *,
    host: str = "127.0.0.1",
    port: int = 4000,
    api_key: Optional[str] = None,
    cors: bool = False,
    fallback_model: Optional[str] = "auto",
    log: Optional[Callable[[str], None]] = None,
) -> FreeLLMServer:
    """Build (but don't start) a server; call ``serve_forever()`` on it."""
    return FreeLLMServer((host, port), llm, api_key=api_key, cors=cors, fallback_model=fallback_model, log=log)


def serve(
    providers: Optional[Sequence[Any]] = None,
    *,
    host: str = "127.0.0.1",
    port: int = 4000,
    api_key: Optional[str] = None,
    cors: bool = False,
    fallback_model: Optional[str] = "auto",
    quiet: bool = False,
    **client_kw: Any,
) -> None:
    """Run ``freelm serve`` in-process until Ctrl+C (providers default to the env)."""
    from .config import providers_from_env

    provs = list(providers) if providers is not None else providers_from_env()
    llm = AsyncFreeLLM(provs, **client_kw)

    def log(line: str) -> None:
        if not quiet:
            print(line, file=sys.stderr, flush=True)

    srv = make_server(llm, host=host, port=port, api_key=api_key, cors=cors, fallback_model=fallback_model, log=log)
    url = url_for(host, srv.server_address[1])
    names = ", ".join(f"{p.name}" + (f" ({len(p.keys)} keys)" if len(p.keys) > 1 else "") for p in provs)
    print(f"freelm {__version__} — OpenAI-compatible endpoint on {url}", file=sys.stderr)
    print(f"providers: {names} · strategy: {llm.strategy}", file=sys.stderr)
    print(f"use it:    base_url={url}  api_key={'<your --api-key>' if api_key else 'anything'}  model=auto",
          file=sys.stderr)
    if host not in _LOOPBACK and not api_key:
        print("warning: listening beyond localhost without --api-key — anyone who can reach this port can spend "
              "your free quota", file=sys.stderr)
    print("Ctrl+C to stop.", file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
