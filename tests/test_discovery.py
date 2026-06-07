import httpx
import pytest
import respx

from conftest import ok_payload

from freelm import FreeLLM, OpenRouter
from freelm.discovery import discover_sync, to_specs

OR_MODELS = "https://openrouter.ai/api/v1/models"
OR_CHAT = "https://openrouter.ai/api/v1/chat/completions"

MODELS_JSON = {
    "data": [
        {"id": "vendor/big-70b:free", "context_length": 131072, "supported_parameters": ["tools", "reasoning"]},
        {"id": "vendor/small-8b:free", "context_length": 8192, "supported_parameters": []},
        {"id": "vendor/paid-70b", "context_length": 1000},  # not free -> filtered
        {
            "id": "vendor/img-only:free",
            "context_length": 4096,
            "architecture": {"output_modalities": ["image"]},  # not text -> filtered
        },
    ]
}


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FREELM_CACHE_DIR", str(tmp_path))


def test_to_specs_filters_and_tags():
    specs = to_specs(MODELS_JSON["data"], free_only=True)
    ids = [s.id for s in specs]
    assert ids == ["vendor/big-70b:free", "vendor/small-8b:free"]  # large first, paid/img dropped
    big = specs[0]
    assert "large" in big.tags and "tools" in big.tags and "reasoning" in big.tags
    small = specs[1]
    assert "small" in small.tags and "fast" in small.tags


@respx.mock
def test_discover_sync_replaces_models_and_caches():
    route = respx.get(OR_MODELS).mock(return_value=httpx.Response(200, json=MODELS_JSON))
    p = OpenRouter("k")
    assert discover_sync(p) is True
    assert [m.id for m in p.models] == ["vendor/big-70b:free", "vendor/small-8b:free"]

    # second provider hits the disk cache, not the network
    p2 = OpenRouter("k")
    assert discover_sync(p2) is True
    assert route.call_count == 1
    assert [m.id for m in p2.models] == ["vendor/big-70b:free", "vendor/small-8b:free"]


@respx.mock
def test_discover_falls_back_to_defaults_on_error():
    respx.get(OR_MODELS).mock(return_value=httpx.Response(500))
    p = OpenRouter("k")
    before = [m.id for m in p.models]
    assert discover_sync(p) is False
    assert [m.id for m in p.models] == before  # untouched hardcoded fallback


@respx.mock
def test_client_discovers_then_chats():
    respx.get(OR_MODELS).mock(return_value=httpx.Response(200, json=MODELS_JSON))
    respx.post(OR_CHAT).mock(return_value=httpx.Response(200, json=ok_payload("hi")))
    with FreeLLM([OpenRouter("k")]) as llm:
        r = llm.chat("hello")
        # discovery ran -> provider now serves the live free list
        assert [m.id for m in llm.providers[0].models][0] == "vendor/big-70b:free"
    assert r.text == "hi"
