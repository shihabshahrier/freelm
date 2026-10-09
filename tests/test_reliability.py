"""Failure modes seen against the live free tiers (2026-10): retired models
(NIM 410 / Gemini 404), per-model quotas and overloads, guard/routing of
concrete ids, error bodies delivered with HTTP 200, and self-explaining
exhaustion errors."""
import asyncio
import json

import httpx
import pytest
import respx

from conftest import ok_payload
from freelm import (
    NIM,
    AsyncFreeLLM,
    ConfigError,
    FreeLLM,
    GoogleAIStudio,
    ModelNotFound,
    ModelSpec,
    NoProvidersAvailable,
    OpenRouter,
)
from freelm.errors import AuthError, RateLimited, Transient, classify

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
G_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

GONE = json.dumps(
    {"status": 410, "title": "Gone", "detail": "The model 'meta/llama-3.3-70b-instruct' has reached its end of life"}
)


def _model_of(request: httpx.Request) -> str:
    return json.loads(request.content)["model"]


def _nim(models):
    return NIM("nvapi-test", models=[ModelSpec(m, ("chat",)) for m in models])


# -- classification ------------------------------------------------------------


@pytest.mark.parametrize(
    "status,body,gone",
    [
        (404, "not found", True),
        (410, GONE, True),
        (400, '{"error":{"message":"The model `x` has been decommissioned","code":"model_decommissioned"}}', True),
        (400, "This model's maximum context length is 8192 tokens", False),
        (413, "Request too large for model llama on tokens per minute", False),
        (422, "model: field required", False),
    ],
)
def test_classify_model_errors(status, body, gone):
    err = classify(status, {}, body, "p")
    assert isinstance(err, ModelNotFound)
    assert err.gone is gone


def test_classify_parses_google_retry_delay():
    body = '{"error":{"code":429,"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"36s"}]}}'
    err = classify(429, {}, body, "google")
    assert isinstance(err, RateLimited)
    assert err.retry_after == 36.0


def test_classify_header_retry_after_wins_over_body():
    err = classify(429, {"retry-after": "5"}, '"retryDelay": "36s"', "google")
    assert err.retry_after == 5.0


# -- retired models (the 2026-08 NIM end-of-life incident) -------------------


@respx.mock
def test_410_gone_fails_over_instead_of_raising():
    respx.post(NIM_URL).mock(return_value=httpx.Response(410, text=GONE))
    respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("from-google")))
    with FreeLLM([NIM("nvapi-test"), GoogleAIStudio("k")], strategy="priority") as llm:
        r = llm.chat("hello")
    assert r.provider == "google"
    nim = llm.providers[0]
    assert nim.keys[0].disabled is False  # a dead model is not a dead key
    assert nim.models[0].id in nim._model_until  # ...but the model is benched


@respx.mock
def test_retired_model_is_benched_across_calls():
    nim = respx.post(NIM_URL).mock(
        side_effect=lambda req: httpx.Response(410, text=GONE)
        if _model_of(req) == "dead/model"
        else httpx.Response(200, json=ok_payload("alive", model=_model_of(req)))
    )
    with FreeLLM([_nim(["dead/model", "alive/model"])]) as llm:
        assert llm.chat("one").text == "alive"
        assert llm.chat("two").text == "alive"
    models = [_model_of(c.request) for c in nim.calls]
    assert models == ["dead/model", "alive/model", "alive/model"]  # dead one tried once only


@respx.mock
def test_dead_models_do_not_starve_the_interleave():
    # NIM lists four retired models; the healthy provider must be reached on
    # the 2nd attempt, not after NIM burns through all of them.
    nim = respx.post(NIM_URL).mock(return_value=httpx.Response(410, text=GONE))
    respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("g")))
    with FreeLLM([_nim(["a", "b", "c", "d"]), GoogleAIStudio("k")], strategy="priority") as llm:
        r = llm.chat("hello")
    assert r.provider == "google"
    assert nim.call_count == 1


@respx.mock
def test_410_during_stream_fails_over():
    sse = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
    respx.post(NIM_URL).mock(return_value=httpx.Response(410, text=GONE))
    respx.post(G_URL).mock(return_value=httpx.Response(200, text=sse))
    with FreeLLM([NIM("nvapi-test"), GoogleAIStudio("k")]) as llm:
        assert "".join(llm.stream("hi")) == "ok"


# -- concrete ids + the OpenRouter free guard ---------------------------------


@respx.mock
def test_known_google_id_is_not_blocked_by_openrouter_guard():
    g = respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("g")))
    orr = respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("or")))
    with FreeLLM([OpenRouter("k", discover=False), GoogleAIStudio("k2")]) as llm:
        r = llm.chat("hi", model="gemini-2.5-flash")
    assert r.provider == "google"
    assert g.call_count == 1 and orr.call_count == 0
    assert _model_of(g.calls[0].request) == "gemini-2.5-flash"


@respx.mock
def test_concrete_id_routes_only_to_providers_that_list_it():
    respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("g")))
    orr = respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("or")))
    or_models = [ModelSpec("vendor/m:free", ("chat",))]
    with FreeLLM([GoogleAIStudio("k2"), OpenRouter("k", discover=False, models=or_models)]) as llm:
        r = llm.chat("hi", model="vendor/m:free")
    assert r.provider == "openrouter"
    assert orr.call_count == 1


@respx.mock
def test_unknown_paid_id_skips_guarded_provider_but_reaches_others():
    g = respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("g")))
    with FreeLLM([OpenRouter("k", discover=False), GoogleAIStudio("k2")]) as llm:
        r = llm.chat("hi", model="some-vendor/unlisted-model")
    assert r.provider == "google"
    assert g.call_count == 1


def test_guard_still_raises_when_nothing_else_can_serve():
    with FreeLLM([OpenRouter("k", discover=False)]) as llm:
        with pytest.raises(ConfigError):
            llm.chat("hi", model="openai/gpt-5")


@respx.mock
def test_guard_skips_one_alias_of_a_fallback_chain():
    route = respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("ok")))
    with FreeLLM([OpenRouter("k", discover=False)]) as llm:
        r = llm.chat("hi", model=["openai/gpt-5", "chat"])
    assert r.text == "ok"
    assert _model_of(route.calls[0].request).endswith(":free")


# -- Google: per-model quotas and overloads -------------------------------------


def _google_two_models():
    return GoogleAIStudio("k", models=[ModelSpec("m1", ("chat",)), ModelSpec("m2", ("chat",))])


@respx.mock
def test_google_429_is_model_scoped():
    body = '{"error":{"code":429,"message":"Quota exceeded for metric generate_content_free_tier_requests, model: m1"}}'
    route = respx.post(G_URL).mock(
        side_effect=lambda req: httpx.Response(429, text=body)
        if _model_of(req) == "m1"
        else httpx.Response(200, json=ok_payload("m2-ok"))
    )
    with FreeLLM([_google_two_models()]) as llm:
        assert llm.chat("one").text == "m2-ok"
        key = llm.providers[0].keys[0]
        assert key.cooldown_until == 0.0  # the key was not cooled
        assert llm.chat("two").text == "m2-ok"
    assert [_model_of(c.request) for c in route.calls] == ["m1", "m2", "m2"]  # m1 benched for call two


@respx.mock
def test_google_503_high_demand_benches_only_the_model():
    body = '{"error":{"code":503,"message":"This model is currently experiencing high demand."}}'
    respx.post(G_URL).mock(
        side_effect=lambda req: httpx.Response(503, text=body)
        if _model_of(req) == "m1"
        else httpx.Response(200, json=ok_payload("m2-ok"))
    )
    with FreeLLM([_google_two_models()]) as llm:
        assert llm.chat("hi").text == "m2-ok"
        key = llm.providers[0].keys[0]
        assert key.cooldown_until == 0.0
        assert key.breaker.failures == 0


@respx.mock
def test_generic_503_still_cools_the_key():
    respx.post(G_URL).mock(return_value=httpx.Response(503, text="upstream connect error"))
    llm = FreeLLM([_google_two_models()])
    with pytest.raises(NoProvidersAvailable):
        llm.chat("hi")
    assert llm.providers[0].keys[0].cooldown_until > 0
    llm.close()


# -- errors delivered with HTTP 200 ----------------------------------------------


@respx.mock
def test_200_with_invalid_json_fails_over():
    respx.post(OR_URL).mock(return_value=httpx.Response(200, text="<html>bad gateway</html>"))
    respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("g")))
    with FreeLLM([OpenRouter("k", discover=False), GoogleAIStudio("k2")]) as llm:
        assert llm.chat("hi").provider == "google"


@respx.mock
def test_200_with_error_object_fails_over():
    respx.post(OR_URL).mock(return_value=httpx.Response(200, json={"error": {"code": 429, "message": "slow down"}}))
    respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("g")))
    with FreeLLM([OpenRouter("k", discover=False), GoogleAIStudio("k2")]) as llm:
        assert llm.chat("hi").provider == "google"


@respx.mock
def test_error_frame_before_first_token_fails_over():
    bad = 'data: {"error":{"code":502,"message":"upstream died"}}\n\n'
    good = 'data: {"choices":[{"delta":{"content":"fine"}}]}\n\ndata: [DONE]\n\n'
    respx.post(OR_URL).mock(return_value=httpx.Response(200, text=bad))
    respx.post(G_URL).mock(return_value=httpx.Response(200, text=good))
    with FreeLLM([OpenRouter("k", discover=False), GoogleAIStudio("k2")]) as llm:
        assert "".join(llm.stream("hi")) == "fine"


@respx.mock
def test_error_frame_after_first_token_raises():
    sse = (
        'data: {"choices":[{"delta":{"content":"par"}}]}\n\n'
        'data: {"error":{"code":500,"message":"upstream died"}}\n\n'
    )
    respx.post(OR_URL).mock(return_value=httpx.Response(200, text=sse))
    respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("g")))
    with FreeLLM([OpenRouter("k", discover=False), GoogleAIStudio("k2")]) as llm:
        got = []
        with pytest.raises(Transient):
            for c in llm.stream("hi"):
                got.append(c)
    assert got == ["par"]  # no mid-stream switch: partial output is not silently spliced


# -- raw chunk streaming ------------------------------------------------------------

TOOL_SSE = (
    'data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
    'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t1",'
    '"function":{"name":"get_weather","arguments":"{\\"city\\":"}}]}}]}\n\n'
    'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
    '"function":{"arguments":"\\"Paris\\"}"}}]}}]}\n\n'
    'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
    "data: [DONE]\n\n"
)


@respx.mock
def test_stream_chunks_keeps_tool_calls_and_finish_reason():
    respx.post(OR_URL).mock(return_value=httpx.Response(200, text=TOOL_SSE))
    with FreeLLM([OpenRouter("k", discover=False)]) as llm:
        chunks = list(llm.stream_chunks("weather?", tools=[{"type": "function"}]))
    assert len(chunks) == 4
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}  # preamble is delivered, in order
    args = "".join(c["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] for c in chunks[1:3])
    assert json.loads(args) == {"city": "Paris"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


@respx.mock
def test_stream_chunks_preamble_is_held_back_during_failover():
    bad = 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\ndata: {"error":{"code":503,"message":"x"}}\n\n'
    respx.post(OR_URL).mock(return_value=httpx.Response(200, text=bad))
    respx.post(G_URL).mock(return_value=httpx.Response(200, text=TOOL_SSE))
    with FreeLLM([OpenRouter("k", discover=False), GoogleAIStudio("k2")]) as llm:
        chunks = list(llm.stream_chunks("hi"))
    assert len(chunks) == 4  # only the healthy provider's stream, no orphaned preamble


def test_astream_chunks():
    @respx.mock
    async def run():
        respx.post(OR_URL).mock(return_value=httpx.Response(200, text=TOOL_SSE))
        async with AsyncFreeLLM([OpenRouter("k", discover=False)]) as llm:
            return [c async for c in llm.astream_chunks("hi")]

    chunks = asyncio.run(run())
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


# -- exhaustion -----------------------------------------------------------------------


@respx.mock
def test_wait_mode_gives_up_promptly_when_waiting_cannot_help():
    # every model benched but the key is ready: waiting can't produce a
    # candidate, so the call must fail fast instead of spinning to the deadline
    respx.post(NIM_URL).mock(return_value=httpx.Response(410, text=GONE))
    llm = FreeLLM([_nim(["a"])], wait=True, max_wait=20, timeout=30)
    import time

    t0 = time.monotonic()
    with pytest.raises(NoProvidersAvailable):
        llm.chat("hi")
    assert time.monotonic() - t0 < 2
    llm.close()


@respx.mock
def test_exhaustion_message_explains_each_provider():
    respx.post(OR_URL).mock(return_value=httpx.Response(401, text='{"error":{"message":"User not found."}}'))
    respx.post(G_URL).mock(return_value=httpx.Response(429, text="Resource exhausted", headers={"retry-after": "30"}))
    llm = FreeLLM([OpenRouter("k", discover=False), GoogleAIStudio("k2", models=[ModelSpec("m1", ("chat",))])])
    with pytest.raises(NoProvidersAvailable) as ei:
        llm.chat("hi")
    msg = str(ei.value)
    assert "openrouter: disabled (auth:401 — key invalid/expired?)" in msg
    assert "freelm doctor" in msg
    assert any(isinstance(e, AuthError) for _, e in ei.value.attempts)
    assert ei.value.status
    llm.close()


# -- SSE edge cases + guard on discovered paid models -----------------------------


@respx.mock
def test_openrouter_midstream_error_shape_with_choices_fails_over():
    # OpenRouter's documented mid-stream error: top-level "error" plus a choice
    # with finish_reason "error" (string code)
    bad = (
        'data: {"id":"x","error":{"code":"server_error","message":"Provider disconnected"},'
        '"choices":[{"index":0,"delta":{"content":""},"finish_reason":"error"}]}\n\n'
    )
    good = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
    respx.post(OR_URL).mock(return_value=httpx.Response(200, text=bad))
    respx.post(G_URL).mock(return_value=httpx.Response(200, text=good))
    with FreeLLM([OpenRouter("k", discover=False), GoogleAIStudio("k2")]) as llm:
        assert "".join(llm.stream("hi")) == "ok"


@respx.mock
def test_sse_decoder_handles_comments_multiline_and_cr():
    sse = (
        ": keep-alive\r\r"
        'data: {"choices":[{"delta":\r'
        'data: {"content":"A"}}]}\r\r'
        'event: message\ndata:{"choices":[{"delta":{"content":"B"}}]}\n\n'
        "data: [DONE]\n\n"
        'data: {"choices":[{"delta":{"content":"after-done"}}]}\n\n'
    )
    respx.post(OR_URL).mock(return_value=httpx.Response(200, text=sse))
    with FreeLLM([OpenRouter("k", discover=False)]) as llm:
        assert "".join(llm.stream("hi")) == "AB"


@respx.mock
def test_free_only_alias_never_resolves_to_discovered_paid_model():
    models = {
        "data": [
            {"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
            {"id": "vendor/free-model:free", "pricing": {"prompt": "0", "completion": "0"}},
        ]
    }
    respx.get("https://openrouter.ai/api/v1/models").mock(return_value=httpx.Response(200, json=models))
    route = respx.post(OR_URL).mock(return_value=httpx.Response(503))
    # discover_free_only=False lists paid models, but free_only (default) must keep `auto` off them
    llm = FreeLLM([OpenRouter("k", discover_free_only=False)])
    with pytest.raises(NoProvidersAvailable):
        llm.chat("hi")
    assert {_model_of(c.request) for c in route.calls} == {"vendor/free-model:free"}
    llm.close()
