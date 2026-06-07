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
        {"id": "vendor/big-70b:free", "context_length": 131072, "supported_parameters": ["tools"]},
        {"id": "vendor/small-8b:free", "context_length": 8192, "supported_parameters": []},
        {"id": "vendor/think-70b:free", "context_length": 200000, "supported_parameters": ["tools", "reasoning"]},
        {"id": "vendor/paid-70b", "context_length": 1000},  # not free -> filtered
        {
            "id": "vendor/img-only:free",
            "context_length": 4096,
            "architecture": {"output_modalities": ["image"]},  # not text -> filtered
        },
    ]
}

# expected order: plain large instruct first, then small, reasoning deprioritized last
EXPECTED_ORDER = ["vendor/big-70b:free", "vendor/small-8b:free", "vendor/think-70b:free"]


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FREELM_CACHE_DIR", str(tmp_path))


def test_to_specs_excludes_non_chat_models():
    data = [
        {"id": "vendor/chat-70b", "context_length": 8192},
        {"id": "whisper-large-v3", "context_length": 0},
        {"id": "vendor/text-embedding-3", "context_length": 0},
        {"id": "playai-tts", "context_length": 0},
        {"id": "vendor/llama-guard-8b", "context_length": 8192},
    ]
    ids = [s.id for s in to_specs(data, free_only=False)]
    assert ids == ["vendor/chat-70b"]  # audio/embed/tts/guard dropped


def test_to_specs_detects_reasoning_by_name():
    # providers whose /models lacks metadata (Groq/Cerebras) — detect by id
    data = [
        {"id": "vendor/gpt-oss-120b", "context_length": 8192},
        {"id": "vendor/llama-3.3-70b", "context_length": 8192},
    ]
    specs = to_specs(data, free_only=False)
    assert specs[0].id == "vendor/llama-3.3-70b"  # plain instruct leads
    assert specs[-1].id == "vendor/gpt-oss-120b" and "reasoning" in specs[-1].tags


def test_to_specs_filters_and_tags():
    specs = to_specs(MODELS_JSON["data"], free_only=True)
    ids = [s.id for s in specs]
    assert ids == EXPECTED_ORDER  # paid + image-only dropped; reasoning sorted last
    big = specs[0]
    assert "large" in big.tags and "tools" in big.tags and "reasoning" not in big.tags
    small = next(s for s in specs if s.id == "vendor/small-8b:free")
    assert "small" in small.tags and "fast" in small.tags
    assert "reasoning" in specs[-1].tags  # reasoning models deprioritized for `auto`


@respx.mock
def test_discover_sync_replaces_models_and_caches():
    route = respx.get(OR_MODELS).mock(return_value=httpx.Response(200, json=MODELS_JSON))
    p = OpenRouter("k")
    assert discover_sync(p) is True
    assert [m.id for m in p.models] == EXPECTED_ORDER

    # second provider hits the disk cache, not the network
    p2 = OpenRouter("k")
    assert discover_sync(p2) is True
    assert route.call_count == 1
    assert [m.id for m in p2.models] == EXPECTED_ORDER


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
