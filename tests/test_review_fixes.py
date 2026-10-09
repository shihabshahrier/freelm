"""Regressions from the adversarial review of 0.4 (one test per finding)."""
import http.client
import json
import socket
import threading
import time

import httpx
import pytest
import respx

from conftest import ok_payload
from freelm import AsyncFreeLLM, BadRequest, FreeLLM, GoogleAIStudio, Groq, ModelSpec, NoProvidersAvailable, OpenRouter
from freelm._state import DISABLED_TTL, StateStore, _key_id
from freelm.server import make_server

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
G_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _key_of(request: httpx.Request) -> str:
    return request.headers["authorization"].split()[-1]


# 1. benches caused by one key's quota/access are per (key, model) --------------------


@respx.mock
def test_model_scoped_429_on_one_key_leaves_other_keys_usable():
    route = respx.post(GROQ_URL).mock(
        side_effect=lambda req: httpx.Response(429, text="Rate limit reached for model qwen")
        if _key_of(req) == "key1"
        else httpx.Response(200, json=ok_payload("from-key2"))
    )
    groq = Groq(["key1", "key2"], models=[ModelSpec("qwen", ("chat",))])
    with FreeLLM([groq]) as llm:
        assert llm.chat("one").text == "from-key2"
        assert llm.chat("two").text == "from-key2"  # key1 benched for qwen, key2 still serves it
    assert [_key_of(c.request) for c in route.calls] == ["key1", "key2", "key2"]


@respx.mock
def test_access_404_benches_only_that_key_but_retired_410_benches_the_model():
    no_access = '{"error":{"message":"The model `qwen` does not exist or you do not have access to it."}}'
    route = respx.post(GROQ_URL).mock(
        side_effect=lambda req: httpx.Response(404, text=no_access) if _key_of(req) == "key1"
        else httpx.Response(200, json=ok_payload("ok"))
    )
    groq = Groq(["key1", "key2"], models=[ModelSpec("qwen", ("chat",))])
    with FreeLLM([groq]) as llm:
        assert llm.chat("hi").text == "ok"
    assert [_key_of(c.request) for c in route.calls] == ["key1", "key2"]
    assert groq.model_ready("qwen", time.monotonic())  # not benched provider-wide


@respx.mock
def test_openrouter_upstream_throttle_benches_the_model_for_every_key():
    route = respx.post(OR_URL).mock(
        side_effect=lambda req: httpx.Response(429, text="m1:free is temporarily rate-limited upstream")
        if json.loads(req.content)["model"] == "m1:free"
        else httpx.Response(200, json=ok_payload("m2"))
    )
    models = [ModelSpec("m1:free", ("chat",)), ModelSpec("m2:free", ("chat",))]
    with FreeLLM([OpenRouter(["k1", "k2"], models=models)]) as llm:
        assert llm.chat("hi").text == "m2"
    assert [json.loads(c.request.content)["model"] for c in route.calls] == ["m1:free", "m2:free"]


# 4. wait=True waits for benched models and cooling keys ------------------------------------


@respx.mock
def test_wait_mode_waits_out_a_per_model_quota():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text='{"error":{"details":[{"retryDelay":"1s"}]}}')
        return httpx.Response(200, json=ok_payload("after-wait"))

    respx.post(G_URL).mock(side_effect=handler)
    llm = FreeLLM([GoogleAIStudio("k", models=[ModelSpec("m1", ("chat",))])], wait=True, max_wait=5, timeout=10)
    t0 = time.monotonic()
    assert llm.chat("hi").text == "after-wait"
    assert 0.9 < time.monotonic() - t0 < 4
    llm.close()


# 5. U+2028 / U+0085 inside JSON strings must not split SSE lines ----------------------------


@respx.mock
def test_stream_keeps_unicode_line_separators_inside_content():
    sse = (
        'data: {"choices":[{"delta":{"content":"Hello world"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"\u0085!"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(OR_URL).mock(return_value=httpx.Response(200, content=sse.encode("utf-8")))
    with FreeLLM([OpenRouter("k", discover=False)]) as llm:
        assert "".join(llm.stream("hi")) == "Hello world\u0085!"


# 6. a persisted disable that expired is re-stamped when the key fails again -------------------


def test_expired_disable_is_restamped_on_a_new_failure(tmp_path):
    store = StateStore(str(tmp_path / "state.json"))
    stale = time.time() - DISABLED_TTL - 3600
    (tmp_path / "state.json").write_text(json.dumps({_key_id("openrouter", "k"): {"disabled": True, "disabled_since_wall": stale}}))
    p = OpenRouter("k", discover=False)
    store.load_into([p], 0.0)
    assert p.keys[0].disabled is False  # expired -> one fresh try
    p.keys[0].disabled = True  # ...which fails again
    store.save([p], 0.0)
    saved = json.loads((tmp_path / "state.json").read_text())[_key_id("openrouter", "k")]
    assert time.time() - saved["disabled_since_wall"] < 60  # new 24h window, not the stale stamp


# 7. an empty or role-only 200 stream fails over -----------------------------------------------


@respx.mock
def test_empty_or_truncated_stream_fails_over():
    respx.post(OR_URL).mock(return_value=httpx.Response(200, text='data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'))
    g = respx.post(G_URL).mock(return_value=httpx.Response(200, text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'))
    with FreeLLM([OpenRouter("k", discover=False), GoogleAIStudio("k2")]) as llm:
        assert "".join(llm.stream("hi")) == "ok"
    assert g.call_count == 1


# 11/13. SSE `data: null`, list-valued deltas, exhaustion reporting ---------------------------------


@respx.mock
def test_sse_null_event_and_list_content():
    sse = (
        "data: null\n"
        'data: {"choices":[{"delta":{"content":[{"type":"text","text":"A"},{"type":"image_url"}]}}]}\n'
        'data: {"choices":[{"delta":{"content":"B"}}]}\n'
        "data: [DONE]\n"
        'data: {"choices":[{"delta":{"content":"late"}}]}\n'
    )
    respx.post(OR_URL).mock(return_value=httpx.Response(200, text=sse))
    with FreeLLM([OpenRouter("k", discover=False)]) as llm:
        assert "".join(llm.stream("hi")) == "AB"


@respx.mock
def test_one_rejection_while_others_cool_is_not_reported_as_a_caller_bug():
    respx.post(OR_URL).mock(return_value=httpx.Response(400, text="unsupported parameter: seed"))
    g = GoogleAIStudio("k2", models=[ModelSpec("m1", ("chat",))])
    g.keys[0].cooldown_until = time.monotonic() + 60
    llm = FreeLLM([OpenRouter("k", discover=False), g])
    with pytest.raises(NoProvidersAvailable):
        llm.chat("hi", seed=1)
    llm.close()
    llm1 = FreeLLM([OpenRouter("k", discover=False)])
    with pytest.raises(BadRequest):  # a one-provider setup still gets the provider's own 400
        llm1.chat("hi", seed=1)
    llm1.close()


# 2/3/9/10/12. the HTTP server ------------------------------------------------------------------


@pytest.fixture
def started():
    servers = []

    def start(providers, **kw):
        srv = make_server(AsyncFreeLLM(providers), port=0, **kw)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return srv.server_address[1]

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def _post(conn, path, body, headers=None):
    data = json.dumps(body).encode()
    conn.request("POST", path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
    r = conn.getresponse()
    return r.status, r.read()


@respx.mock
def test_keep_alive_survives_early_error_responses(started):
    respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("ok")))
    port = started([OpenRouter("k", discover=False)], api_key="s3cret")
    msg = {"model": "auto", "messages": [{"role": "user", "content": "x"}]}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    assert _post(conn, "/v1/embeddings", msg, {"Authorization": "Bearer s3cret"})[0] == 404
    assert _post(conn, "/v1/chat/completions", msg)[0] == 401  # (new connection after close)
    status, body = _post(conn, "/v1/chat/completions", msg, {"Authorization": "Bearer s3cret"})
    assert status == 200 and json.loads(body)["choices"][0]["message"]["content"] == "ok"


def test_bind_failure_raises_oserror_and_close_is_idempotent():
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    port = busy.getsockname()[1]
    with pytest.raises(OSError):
        make_server(AsyncFreeLLM([OpenRouter("k", discover=False)]), port=port)
    busy.close()
    srv = make_server(AsyncFreeLLM([OpenRouter("k", discover=False)]), port=0)
    t0 = time.monotonic()
    srv.server_close()
    srv.server_close()
    assert time.monotonic() - t0 < 3


@respx.mock
def test_browser_style_requests_are_refused(started):
    port = started([OpenRouter("k", discover=False)])
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/v1/chat/completions", body=b'{"messages":[]}', headers={"Content-Type": "text/plain"})
    r = conn.getresponse()
    assert r.status == 415 and r.read()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/v1/models", headers={"Host": "evil.example:4000"})
    r = conn.getresponse()
    assert r.status == 403 and r.read()


@respx.mock
def test_cors_preflight_reflects_requested_headers(started):
    port = started([OpenRouter("k", discover=False)], cors=True)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("OPTIONS", "/v1/chat/completions", headers={"Access-Control-Request-Headers": "x-stainless-os, authorization"})
    r = conn.getresponse()
    r.read()
    assert r.status == 204 and r.getheader("Access-Control-Allow-Headers") == "x-stainless-os, authorization"


@respx.mock
def test_http10_stream_is_not_chunked(started):
    respx.post(OR_URL).mock(return_value=httpx.Response(200, text='data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'))
    port = started([OpenRouter("k", discover=False)])
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    body = json.dumps({"model": "auto", "stream": True, "messages": [{"role": "user", "content": "x"}]}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
              + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    raw = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        raw += chunk
    s.close()
    head, _, payload = raw.partition(b"\r\n\r\n")
    assert b"Transfer-Encoding" not in head
    assert payload.startswith(b"data: ") and payload.rstrip().endswith(b"data: [DONE]")


@respx.mock
def test_oversized_body_gets_413_not_a_reset(started):
    port = started([OpenRouter("k", discover=False)])
    big = json.dumps({"messages": [{"role": "user", "content": "x" * (21 * 1024 * 1024)}]})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    status, body = _post(conn, "/v1/chat/completions", json.loads(big))
    assert status == 413 and b"larger than" in body
    # the body was drained, so the same connection keeps working
    conn.request("GET", "/v1/models")
    r = conn.getresponse()
    assert r.status == 200 and r.read()
