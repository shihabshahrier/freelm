"""`freelm serve`: a real localhost socket, upstream providers mocked with respx
(the test client uses urllib, which respx doesn't intercept)."""
import json
import threading
import urllib.error
import urllib.request

import httpx
import pytest
import respx

from conftest import ok_payload
from freelm import AsyncFreeLLM, GoogleAIStudio, OpenRouter
from freelm.server import make_server

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
G_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


@pytest.fixture
def server():
    started = []

    def start(providers, **kw):
        srv = make_server(AsyncFreeLLM(providers), port=0, **kw)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield start
    for srv in started:
        srv.shutdown()
        srv.server_close()


def _req(url, body=None, headers=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=h, method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


@respx.mock
def test_chat_completion_json(server):
    respx.post(OR_URL).mock(return_value=httpx.Response(429, text="slow down"))
    respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("pong", model="gemini-x")))
    base = server([OpenRouter("k", discover=False), GoogleAIStudio("k2")])
    status, headers, text = _req(base + "/v1/chat/completions",
                                 {"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    body = json.loads(text)
    assert status == 200
    assert body["object"] == "chat.completion" and body["choices"][0]["message"]["content"] == "pong"
    assert headers["X-FreeLLM-Provider"] == "google"


@respx.mock
def test_streaming_sse_passes_chunks_and_done(server):
    sse = (
        'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(OR_URL).mock(return_value=httpx.Response(200, text=sse))
    base = server([OpenRouter("k", discover=False)])
    status, headers, text = _req(base + "/v1/chat/completions",
                                 {"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200 and headers["Content-Type"] == "text/event-stream"
    events = [line[6:] for line in text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks) == "Hello"
    assert len({c["id"] for c in chunks}) == 1 and chunks[-1]["choices"][0]["finish_reason"] == "stop"


@respx.mock
def test_errors_map_to_openai_shapes(server):
    respx.post(OR_URL).mock(return_value=httpx.Response(401, text="bad key"))
    base = server([OpenRouter("k", discover=False)])
    status, _, text = _req(base + "/v1/chat/completions", {"model": "auto", "messages": [{"role": "user", "content": "x"}]})
    assert status == 503
    assert json.loads(text)["error"]["code"] == "no_providers_available"
    status, _, text = _req(base + "/v1/chat/completions", {"model": "auto"})
    assert status == 400 and "messages" in json.loads(text)["error"]["message"]
    status, _, _ = _req(base + "/v1/nope", {"x": 1})
    assert status == 404


@respx.mock
def test_unknown_model_falls_back_to_auto(server):
    route = respx.post(G_URL).mock(
        side_effect=lambda req: httpx.Response(404, text="model not found")
        if json.loads(req.content)["model"] == "gpt-4o"
        else httpx.Response(200, json=ok_payload("ok"))
    )
    base = server([GoogleAIStudio("k2")])
    status, _, text = _req(base + "/v1/chat/completions", {"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]})
    assert status == 200 and json.loads(text)["choices"][0]["message"]["content"] == "ok"
    assert [json.loads(c.request.content)["model"] for c in route.calls][0] == "gpt-4o"


@respx.mock
def test_api_key_is_enforced_but_health_is_open(server):
    respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("ok")))
    base = server([OpenRouter("k", discover=False)], api_key="s3cret")
    body = {"model": "auto", "messages": [{"role": "user", "content": "x"}]}
    assert _req(base + "/v1/chat/completions", body)[0] == 401
    assert _req(base + "/v1/chat/completions", body, {"Authorization": "Bearer wrong"})[0] == 401
    assert _req(base + "/v1/chat/completions", body, {"Authorization": "Bearer s3cret"})[0] == 200
    assert _req(base + "/health")[0] == 200
    assert _req(base + "/v1/models")[0] == 401


@respx.mock
def test_models_lists_aliases_then_provider_models(server):
    base = server([OpenRouter("k", discover=False)])
    status, _, text = _req(base + "/v1/models")
    ids = [m["id"] for m in json.loads(text)["data"]]
    assert status == 200 and ids[:1] == ["auto"] and any(i.endswith(":free") for i in ids)


@respx.mock
def test_cors_preflight(server):
    base = server([OpenRouter("k", discover=False)], cors=True)
    status, headers, _ = _req(base + "/v1/chat/completions", method="OPTIONS")
    assert status == 204 and headers["Access-Control-Allow-Origin"] == "*"
