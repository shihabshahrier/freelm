import asyncio

import httpx
import respx

from freelm import AsyncFreeLLM, FreeLLM, GoogleAIStudio, OpenRouter

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
G_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

SSE = (
    'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
    "data: [DONE]\n\n"
)


@respx.mock
def test_stream_basic():
    respx.post(OR_URL).mock(
        return_value=httpx.Response(200, text=SSE, headers={"content-type": "text/event-stream"})
    )
    with FreeLLM([OpenRouter("k", discover=False)]) as llm:
        out = "".join(llm.stream("hi"))
    assert out == "Hello"


@respx.mock
def test_stream_fails_over_before_first_token():
    respx.post(OR_URL).mock(return_value=httpx.Response(429, text="account rate limit"))
    respx.post(G_URL).mock(return_value=httpx.Response(200, text=SSE))
    with FreeLLM([OpenRouter("k", discover=False), GoogleAIStudio("k2")]) as llm:
        out = "".join(llm.stream("hi"))
    assert out == "Hello"


def test_astream_basic():
    @respx.mock
    async def run():
        respx.post(OR_URL).mock(return_value=httpx.Response(200, text=SSE))
        out = ""
        async with AsyncFreeLLM([OpenRouter("k", discover=False)]) as llm:
            async for c in llm.astream("hi"):
                out += c
        return out

    assert asyncio.run(run()) == "Hello"
