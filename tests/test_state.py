import json

import httpx
import pytest
import respx

from conftest import ok_payload

from freelm import FreeLLM, OpenRouter
from freelm._state import StateStore

OR_URL = "https://openrouter.ai/api/v1/chat/completions"


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FREELM_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_state_roundtrip_carries_quota_and_disable(tmp_path):
    store = StateStore(str(tmp_path / "state.json"))
    p = OpenRouter("sk-or-abc", discover=False)
    k = p.keys[0]
    now = 100.0
    k.rpd_used = 7
    k.rpd_reset = now + 1000
    k.cooldown_until = now + 30
    k.disabled = True
    k.last_error = "auth:401"
    store.save([p], now)

    p2 = OpenRouter("sk-or-abc", discover=False)
    store.load_into([p2], 5.0)  # fresh process: different monotonic origin
    k2 = p2.keys[0]
    assert k2.rpd_used == 7
    assert k2.disabled is True
    assert k2.last_error == "auth:401"
    assert k2.cooldown_until > 5.0
    assert k2.rpd_reset > 5.0


def test_state_file_never_contains_raw_keys(tmp_path):
    store = StateStore(str(tmp_path / "state.json"))
    p = OpenRouter("sk-or-supersecret-key", discover=False)
    store.save([p], 0.0)
    raw = (tmp_path / "state.json").read_text()
    assert "supersecret" not in raw


def test_state_ignores_corrupt_file(tmp_path):
    f = tmp_path / "state.json"
    f.write_text("{not json")
    store = StateStore(str(f))
    p = OpenRouter("k", discover=False)
    store.load_into([p], 0.0)  # must not raise
    assert p.keys[0].rpd_used == 0
    store.save([p], 0.0)  # rewrites cleanly
    assert isinstance(json.loads(f.read_text()), dict)


@respx.mock
def test_client_persist_survives_restart(tmp_path):
    respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("hi")))
    with FreeLLM([OpenRouter("sk-or-x", discover=False)], persist=True) as llm:
        llm.chat("hello")
        used = llm.providers[0].keys[0].rpd_used
        assert used == 1

    # "restart": new client, same env -> state reloaded from disk
    llm2 = FreeLLM([OpenRouter("sk-or-x", discover=False)], persist=True)
    assert llm2.providers[0].keys[0].rpd_used == used
    llm2.close()
