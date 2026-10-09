import httpx
import respx

from conftest import ok_payload
from freelm import FreeLLM, OpenRouter
from freelm.compat import OpenAI

OR_URL = "https://openrouter.ai/api/v1/chat/completions"


@respx.mock
def test_openai_compat_shim():
    respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("compat")))
    client = OpenAI(FreeLLM([OpenRouter("k1")]))
    r = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert r.choices[0].message.content == "compat"
    assert r.model_dump()["choices"][0]["finish_reason"] == "stop"
    client.close()


@respx.mock
def test_openai_compat_accepts_sdk_ctor_kwargs(monkeypatch):
    # real OpenAI users construct with api_key/base_url/etc — must not crash
    for var in (
        "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_KEY", "FREELM_GOOGLE_KEYS",
        "NVIDIA_API_KEY", "NIM_API_KEY", "FREELM_NIM_KEYS", "GROQ_API_KEY", "FREELM_GROQ_KEYS",
        "CEREBRAS_API_KEY", "FREELM_CEREBRAS_KEYS", "MISTRAL_API_KEY", "FREELM_MISTRAL_KEYS",
        "FREELM_OPENROUTER_KEYS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("ok")))
    client = OpenAI(api_key="sk-ignored", base_url="https://api.openai.com/v1", organization="org", max_retries=2)
    r = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}])
    assert r.choices[0].message.content == "ok"
    client.close()


@respx.mock
def test_openai_compat_stream_yields_chunks():
    sse = (
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    respx.post(OR_URL).mock(
        return_value=httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
    )
    client = OpenAI(FreeLLM([OpenRouter("k1", discover=False)]))
    out = ""
    for chunk in client.chat.completions.create(
        model="auto", messages=[{"role": "user", "content": "hi"}], stream=True
    ):
        assert chunk.object == "chat.completion.chunk"
        out += chunk.choices[0].delta.content or ""
    assert out == "Hello"
    client.close()


# -- OpenAI-SDK fidelity ------------------------------------------------------------

TOOL_PAYLOAD = {
    "id": "chatcmpl-1",
    "model": "m",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}


@respx.mock
def test_compat_tool_call_loop_round_trips():
    import json

    route = respx.post(OR_URL).mock(
        side_effect=[httpx.Response(200, json=TOOL_PAYLOAD), httpx.Response(200, json=ok_payload("sunny"))]
    )
    with OpenAI(FreeLLM([OpenRouter("k1", discover=False)])) as client:
        msgs = [{"role": "user", "content": "weather in Paris?"}]
        r = client.chat.completions.create(model="auto", messages=msgs, tools=[{"type": "function"}])
        call = r.choices[0].message.tool_calls[0]
        assert (call.id, call.function.name, json.loads(call.function.arguments)) == ("call_1", "get_weather", {"city": "Paris"})
        assert r.created > 0 and r.object == "chat.completion" and r.usage.total_tokens == 8
        assert r.choices[0].message.refusal is None  # optional SDK fields read as None
        msgs.append(r.choices[0].message)  # the SDK idiom: pass the message object back
        msgs.append({"role": "tool", "tool_call_id": call.id, "content": "sunny"})
        r2 = client.chat.completions.create(model="auto", messages=msgs)
        assert r2.choices[0].message.content == "sunny"
    sent = json.loads(route.calls[1].request.content)["messages"][1]
    assert sent["tool_calls"][0]["id"] == "call_1" and "content" not in sent  # null fields dropped


@respx.mock
def test_compat_dump_helpers_and_extra_body():
    import json

    route = respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("x")))
    client = OpenAI(FreeLLM([OpenRouter("k1", discover=False)]))
    r = client.chat.completions.create(
        model="auto", messages=[{"role": "user", "content": "hi"}],
        extra_body={"top_k": 5}, extra_headers={"X-A": "1"}, timeout=10,
    )
    body = json.loads(route.calls[0].request.content)
    assert body["top_k"] == 5 and "extra_headers" not in body and "timeout" not in body
    assert r.to_dict()["choices"][0]["message"]["content"] == "x"
    assert json.loads(r.model_dump_json())["model"] == "test-model"
    assert "logprobs" not in json.dumps(r.model_dump(exclude_none=True))
    client.close()


@respx.mock
def test_compat_stream_has_tool_deltas_finish_reason_and_context_manager():
    sse = (
        'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"f","arguments":"{}"}}]}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(OR_URL).mock(return_value=httpx.Response(200, text=sse))
    client = OpenAI(FreeLLM([OpenRouter("k1", discover=False)]))
    with client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}], stream=True) as stream:
        chunks = list(stream)
    assert len({c.id for c in chunks}) == 1 and all(c.created for c in chunks)
    assert chunks[1].choices[0].delta.tool_calls[0].function.name == "f"
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
    client.close()


@respx.mock
def test_compat_models_list_has_aliases_and_provider_models():
    client = OpenAI(FreeLLM([OpenRouter("k1", discover=False)]))
    ids = [m.id for m in client.models.list().data]
    assert "auto" in ids and "chat" in ids
    assert any(i.endswith(":free") for i in ids)
    client.close()


def test_async_compat_stream_and_close():
    import asyncio

    from freelm import AsyncFreeLLM
    from freelm.compat import AsyncOpenAI

    @respx.mock
    async def run():
        respx.post(OR_URL).mock(
            return_value=httpx.Response(200, text='data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n')
        )
        async with AsyncOpenAI(AsyncFreeLLM([OpenRouter("k1", discover=False)])) as client:
            stream = await client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "x"}], stream=True)
            async with stream:
                return [c.choices[0].delta.content async for c in stream]

    assert asyncio.run(run()) == ["hi"]
