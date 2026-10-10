"""Failover speed: hedged attempts, smart (latency-aware) routing, parallel
discovery, connect timeout."""
import asyncio
import time

import httpx
import pytest
import respx

from conftest import ok_payload
from freelm import AsyncFreeLLM, FreeLLM, ModelSpec, NoProvidersAvailable, Provider
from freelm import _engine as engine
from freelm.client import CONNECT_TIMEOUT, _attempt_timeout
from freelm.providers.base import LATENCY_PRIOR_MS, LATENCY_TTL
from freelm.strategy import Candidate, order_candidates


def prov(name, **kw):
    return Provider("k-" + name, name=name, base_url=f"https://{name}.test/v1",
                    models=[ModelSpec("m", ("chat",))], rpm=None, **kw)


def url(name):
    return f"https://{name}.test/v1/chat/completions"


SSE_OK = 'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'


# -- pure helpers ---------------------------------------------------------------------


def test_hedge_delay_adapts_to_measured_latency():
    p = prov("a")
    cand = Candidate(p, p.keys[0], "m")
    assert engine.hedge_delay(cand, False, stream=False) is None
    assert engine.hedge_delay(cand, 0, stream=False) is None
    assert engine.hedge_delay(cand, 2.5, stream=False) == 2.5  # fixed
    floor, cap, unmeasured = engine.HEDGE_CHAT
    assert engine.hedge_delay(cand, True, stream=False) == unmeasured
    assert engine.hedge_delay(cand, True, stream=True) == engine.HEDGE_STREAM[2]  # streams hedge sooner
    p.keys[0].ewma_latency = 100.0  # fast key: never below the floor
    assert engine.hedge_delay(cand, True, stream=False) == floor
    p.keys[0].ewma_latency = 2000.0  # 3x its usual latency
    assert engine.hedge_delay(cand, True, stream=False) == 6.0
    p.keys[0].ewma_latency = 60000.0
    assert engine.hedge_delay(cand, True, stream=False) == cap


def test_smart_order_prefers_fast_tries_unknown_and_forgets_old_samples():
    fast, slow, new = prov("fast"), prov("slow"), prov("new")
    now = 1000.0
    for p, ms in ((fast, 300.0), (slow, 9000.0)):
        p.keys[0].ewma_latency, p.keys[0].latency_at = ms, now
    names = lambda: [c.provider.name for c in order_candidates([slow, new, fast], "auto", now, "smart", {})]  # noqa: E731
    assert names() == ["fast", "new", "slow"]  # unknown = typical (LATENCY_PRIOR_MS)
    assert new.expected_latency(now) == LATENCY_PRIOR_MS
    slow.keys[0].latency_at = now - LATENCY_TTL - 1  # stale: slow gets another chance
    assert names() == ["fast", "slow", "new"]  # ties keep the given order
    fast.priority = 1  # priority tiers still come first
    assert names()[-1] == "fast"


def test_connect_timeout_is_short_even_with_a_long_deadline():
    t = _attempt_timeout(60.0, time.monotonic() + 60.0)
    assert t.connect == CONNECT_TIMEOUT and t.read > 50
    assert _attempt_timeout(60.0, time.monotonic() + 2.0).connect <= 2.0


# -- the race ---------------------------------------------------------------------------


@respx.mock
def test_a_slow_provider_is_hedged_and_remembered():
    def slow(req):
        time.sleep(0.6)
        return httpx.Response(200, json=ok_payload("slow"))

    respx.post(url("a")).mock(side_effect=slow)
    fast = respx.post(url("b")).mock(return_value=httpx.Response(200, json=ok_payload("fast")))
    a, b = prov("a"), prov("b")
    events = []
    with FreeLLM([a, b], hedge=0.05, on_event=events.append) as llm:
        t0 = time.monotonic()
        assert llm.chat("hi").text == "fast"
        assert time.monotonic() - t0 < 0.5  # didn't wait for the slow one
        assert [e.kind for e in events[:2]] == ["attempt", "hedge"]
        assert a.keys[0].ewma_latency >= 50  # marked slow ...
        events.clear()
        assert llm.chat("again").text == "fast"  # ... so smart routing goes to b first
        assert [e.provider for e in events if e.kind == "attempt"] == ["b"]
    assert fast.call_count == 2


@respx.mock
def test_hedging_off_waits_for_the_running_attempt():
    def slow(req):
        time.sleep(0.2)
        return httpx.Response(200, json=ok_payload("slow"))

    respx.post(url("a")).mock(side_effect=slow)
    b = respx.post(url("b")).mock(return_value=httpx.Response(200, json=ok_payload("fast")))
    events = []
    with FreeLLM([prov("a"), prov("b")], hedge=False, on_event=events.append) as llm:
        assert llm.chat("hi").text == "slow"
    assert b.call_count == 0 and "hedge" not in [e.kind for e in events]


@respx.mock
def test_a_hedge_failing_fast_keeps_waiting_for_the_original():
    def slow(req):
        time.sleep(0.3)
        return httpx.Response(200, json=ok_payload("slow"))

    respx.post(url("a")).mock(side_effect=slow)
    respx.post(url("b")).mock(return_value=httpx.Response(429, text="limit"))
    with FreeLLM([prov("a"), prov("b")], hedge=0.05) as llm:
        assert llm.chat("hi").text == "slow"


@respx.mock
def test_a_stalled_stream_is_hedged_before_the_first_token():
    def stalled(req):
        time.sleep(0.6)
        return httpx.Response(200, text=SSE_OK.replace("hi", "late"))

    respx.post(url("a")).mock(side_effect=stalled)
    respx.post(url("b")).mock(side_effect=lambda req: httpx.Response(200, text=SSE_OK))
    with FreeLLM([prov("a"), prov("b")], hedge=0.05) as llm:
        t0 = time.monotonic()
        assert "".join(llm.stream("hi")) == "hi"
        assert time.monotonic() - t0 < 0.5
        assert [c["choices"][0]["delta"]["content"] for c in llm.stream_chunks("hi")] == ["hi"]


@respx.mock
def test_an_attempt_that_outlives_the_deadline_is_recorded_as_a_timeout():
    def hang(req):
        time.sleep(1.0)
        return httpx.Response(200, json=ok_payload())

    respx.post(url("a")).mock(side_effect=hang)
    a = prov("a")
    with FreeLLM([a], timeout=0.2) as llm:
        t0 = time.monotonic()
        with pytest.raises(NoProvidersAvailable, match="Transient"):
            llm.chat("hi")
        assert time.monotonic() - t0 < 0.6
    assert a.keys[0].last_error == "transient:0"  # cooled: the next call skips it


@respx.mock
def test_async_hedge_cancels_the_loser():
    cancelled = []

    async def slow(req):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return httpx.Response(200, json=ok_payload("slow"))

    respx.post(url("a")).mock(side_effect=slow)
    respx.post(url("b")).mock(return_value=httpx.Response(200, json=ok_payload("fast")))

    async def main():
        async with AsyncFreeLLM([prov("a"), prov("b")], hedge=0.05) as llm:
            t0 = time.monotonic()
            assert (await llm.chat("hi")).text == "fast"
            assert time.monotonic() - t0 < 0.5
            await asyncio.sleep(0.05)  # let the cancellation land

    asyncio.run(main())
    assert cancelled  # the slow attempt didn't keep running


@respx.mock
def test_async_stalled_stream_is_hedged():
    async def stalled(req):
        await asyncio.sleep(5)
        return httpx.Response(200, text=SSE_OK.replace("hi", "late"))

    respx.post(url("a")).mock(side_effect=stalled)
    respx.post(url("b")).mock(side_effect=lambda req: httpx.Response(200, text=SSE_OK))

    async def main():
        async with AsyncFreeLLM([prov("a"), prov("b")], hedge=0.05) as llm:
            t0 = time.monotonic()
            assert "".join([t async for t in llm.astream("hi")]) == "hi"
            assert time.monotonic() - t0 < 0.5

    asyncio.run(main())


@respx.mock
def test_sync_discovery_runs_in_parallel():
    def catalog(req):
        time.sleep(0.3)
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    respx.get(url__regex=r"https://p\d\.test/v1/models").mock(side_effect=catalog)
    respx.post(url__regex=r"https://p\d\.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_payload()))
    provs = [Provider("k", name=f"p{i}", base_url=f"https://p{i}.test/v1", discover=True) for i in range(4)]
    with FreeLLM(provs) as llm:
        t0 = time.monotonic()
        llm.chat("hi")
        assert time.monotonic() - t0 < 0.9  # 4 x 0.3 s one after another would be 1.2 s
    assert all(p._discovered for p in provs)
