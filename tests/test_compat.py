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
