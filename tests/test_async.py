import asyncio

import httpx
import respx

from conftest import ok_payload

from freelm import AsyncFreeLLM, GoogleAIStudio, OpenRouter

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
G_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def test_async_success():
    @respx.mock
    async def run():
        respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("async-hi")))
        async with AsyncFreeLLM([OpenRouter("k1")]) as llm:
            r = await llm.chat("hello")
        return r

    r = asyncio.run(run())
    assert r.text == "async-hi"
    assert r.provider == "openrouter"


def test_async_failover():
    @respx.mock
    async def run():
        respx.post(OR_URL).mock(return_value=httpx.Response(429))
        respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("g")))
        async with AsyncFreeLLM([OpenRouter("k1"), GoogleAIStudio("k2")]) as llm:
            return await llm.chat("hello")

    r = asyncio.run(run())
    assert r.provider == "google"
