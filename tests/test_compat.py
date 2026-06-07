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
