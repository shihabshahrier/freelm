import httpx
import pytest
import respx

from conftest import ok_payload

from freelm import FreeLLM, GoogleAIStudio, ModelSpec, NIM, NoProvidersAvailable, OpenRouter

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
G_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


@respx.mock
def test_success():
    respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("hi")))
    with FreeLLM([OpenRouter("sk-or-test")]) as llm:
        r = llm.chat("hello")
    assert r.text == "hi"
    assert r.provider == "openrouter"
    assert r.usage.total_tokens == 5


@respx.mock
def test_rotate_key_on_429():
    route = respx.post(OR_URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=ok_payload("second"))]
    )
    with FreeLLM([OpenRouter(["key-a", "key-b"])]) as llm:
        r = llm.chat("hello")
    assert r.text == "second"
    assert route.call_count == 2


@respx.mock
def test_failover_across_providers():
    respx.post(OR_URL).mock(return_value=httpx.Response(429))
    respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("from-google")))
    with FreeLLM([OpenRouter("k1"), GoogleAIStudio("k2")], strategy="priority") as llm:
        r = llm.chat("hello")
    assert r.text == "from-google"
    assert r.provider == "google"


@respx.mock
def test_quota_402_disables_key_and_fails_over():
    # OpenRouter returns 402 when the account is out of credits — that must
    # disable the key and fail over, not abort the whole call.
    respx.post(OR_URL).mock(return_value=httpx.Response(402, text="Insufficient credits"))
    respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("from-google")))
    with FreeLLM([OpenRouter("broke-key"), GoogleAIStudio("k2")], strategy="priority") as llm:
        r = llm.chat("hello")
    assert r.provider == "google"
    key = llm.providers[0].keys[0]
    assert key.disabled is True
    assert key.last_error == "quota:402"


@respx.mock
def test_auth_disables_key_then_exhausts():
    respx.post(OR_URL).mock(return_value=httpx.Response(401, text="invalid key"))
    llm = FreeLLM([OpenRouter("bad-key")])
    with pytest.raises(NoProvidersAvailable):
        llm.chat("hello")
    assert llm.providers[0].keys[0].disabled is True
    llm.close()


@respx.mock
def test_transient_then_recover_via_failover():
    respx.post(OR_URL).mock(return_value=httpx.Response(503))
    respx.post(NIM_URL).mock(return_value=httpx.Response(500))
    respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("ok")))
    with FreeLLM([OpenRouter("k1"), NIM("nvapi-x"), GoogleAIStudio("k2")]) as llm:
        r = llm.chat("hello")
    assert r.provider == "google"
    # the openrouter key took a breaker failure
    assert llm.providers[0].keys[0].breaker.failures >= 1


@respx.mock
def test_bad_request_raises_immediately():
    respx.post(OR_URL).mock(return_value=httpx.Response(400, text="invalid temperature"))
    llm = FreeLLM([OpenRouter("k1"), GoogleAIStudio("k2")])
    with pytest.raises(Exception) as ei:
        llm.chat("hello")
    # not model-related -> treated as caller bug, surfaced directly
    assert "400" in str(ei.value)
    llm.close()


@respx.mock
def test_model_scoped_429_tries_next_model_same_key():
    # OpenRouter throttles one free model upstream; a different model should be
    # tried on the SAME key without benching the key.
    route = respx.post(OR_URL).mock(
        side_effect=[
            httpx.Response(429, text="model X is temporarily rate-limited upstream"),
            httpx.Response(200, json=ok_payload("recovered")),
        ]
    )
    llm = FreeLLM([OpenRouter("only-key")])
    r = llm.chat("hello")
    assert r.text == "recovered"
    assert route.call_count == 2
    key = llm.providers[0].keys[0]
    assert key.cooldown_until == 0.0   # key was NOT cooled (model-scoped 429)
    assert key.disabled is False
    llm.close()


@respx.mock
def test_key_scoped_429_cools_the_key():
    respx.post(OR_URL).mock(return_value=httpx.Response(429, text="account rate limit exceeded"))
    llm = FreeLLM([OpenRouter("only-key")])
    with pytest.raises(NoProvidersAvailable):
        llm.chat("hello")
    assert llm.providers[0].keys[0].cooldown_until > 0.0  # whole key cooled
    llm.close()


@respx.mock
def test_interleave_reaches_second_provider_despite_many_models():
    # provider 1 has MANY models, all model-scoped throttled (key stays hot).
    # breadth-first interleaving must still reach provider 2 quickly.
    respx.post(OR_URL).mock(return_value=httpx.Response(429, text="temporarily rate-limited upstream"))
    respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("google")))
    many = [ModelSpec(f"vendor/m{i}:free", ("chat", "large")) for i in range(10)]
    llm = FreeLLM(
        [OpenRouter("k", models=many, discover=False), GoogleAIStudio("k2")],
        strategy="priority",
    )
    r = llm.chat("hi")
    assert r.provider == "google"   # would starve under old provider-major ordering
    llm.close()


@respx.mock
def test_wait_retries_recovered_key():
    # single key cools for ~1s, then recovers; wait=True should retry it.
    route = respx.post(OR_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "1"}, text="account rate limit"),
            httpx.Response(200, json=ok_payload("recovered")),
        ]
    )
    llm = FreeLLM([OpenRouter("only-key", discover=False)], wait=True, max_wait=3)
    r = llm.chat("hi")
    assert r.text == "recovered"
    assert route.call_count == 2
    llm.close()


@respx.mock
def test_health_report():
    respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload()))
    with FreeLLM([OpenRouter("k1")]) as llm:
        llm.chat("hi")
        h = llm.health()
    assert h[0]["provider"] == "openrouter"
    assert h[0]["breaker"] == "closed"


def test_free_guard_blocks_paid_passthrough():
    from freelm import ConfigError
    import pytest as _pytest

    llm = FreeLLM([OpenRouter("k", discover=False)])
    with _pytest.raises(ConfigError, match="free_only=False"):
        llm.chat("hi", model="openai/gpt-4o")
    llm.close()


@respx.mock
def test_free_guard_opt_out_and_free_ids_pass():
    respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("ok")))
    # :free suffix always passes
    with FreeLLM([OpenRouter("k", discover=False)]) as llm:
        assert llm.chat("hi", model="meta-llama/llama-3.3-70b-instruct:free").text == "ok"
    # free_only=False allows paid ids on the user's own account
    with FreeLLM([OpenRouter("k", discover=False, free_only=False)]) as llm:
        assert llm.chat("hi", model="openai/gpt-4o").text == "ok"


@respx.mock
def test_per_call_model_chain_fails_over_to_second_id():
    route = respx.post(OR_URL).mock(
        side_effect=[
            httpx.Response(429, text="model X is temporarily rate-limited upstream"),
            httpx.Response(200, json=ok_payload("via-second")),
        ]
    )
    llm = FreeLLM([OpenRouter("k", discover=False)])
    r = llm.chat("hi", model=["first/model:free", "second/model:free"])
    assert r.text == "via-second"
    assert route.call_count == 2
    bodies = [c.request.read() for c in route.calls]
    assert b'"first/model:free"' in bodies[0]
    assert b'"second/model:free"' in bodies[1]
    llm.close()


@respx.mock
def test_on_event_sees_failover_sequence_and_survives_bad_callback():
    respx.post(OR_URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=ok_payload("done"))]
    )
    events = []

    def hook(e):
        events.append((e.kind, e.provider))
        raise RuntimeError("callbacks must never break the call")

    with FreeLLM([OpenRouter(["key-a", "key-b"], discover=False)], on_event=hook) as llm:
        assert llm.chat("hi").text == "done"
    kinds = [k for k, _ in events]
    assert kinds == ["attempt", "error", "attempt", "success"]
    assert all(p == "openrouter" for _, p in events)


@respx.mock
def test_tools_and_response_format_reach_request_body():
    payload = ok_payload("calling")
    payload["choices"][0]["message"]["tool_calls"] = [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
    ]
    route = respx.post(OR_URL).mock(return_value=httpx.Response(200, json=payload))
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    with FreeLLM([OpenRouter("k", discover=False)]) as llm:
        r = llm.chat("weather?", tools=tools, tool_choice="auto", response_format={"type": "json_object"})
    body = route.calls[0].request.read()
    assert b'"tools"' in body and b'"tool_choice"' in body and b'"response_format"' in body
    assert r.tool_calls and r.tool_calls[0]["function"]["name"] == "get_weather"
