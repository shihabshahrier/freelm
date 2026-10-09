"""Hardening from the 2026-10 review: error taxonomy edge cases, secret-safe
reprs, persistence correctness, discovery tagging and model-list control."""
import asyncio
import json
import time

import httpx
import pytest
import respx

from conftest import ok_payload
from freelm import (
    AsyncFreeLLM,
    BadRequest,
    Cerebras,
    ConfigError,
    FreeLLM,
    GoogleAIStudio,
    Groq,
    ModelNotFound,
    ModelSpec,
    NoProvidersAvailable,
    OpenRouter,
)
from freelm._state import DISABLED_TTL, StateStore
from freelm.discovery import _size_tags, to_specs
from freelm.errors import AuthError, RateLimited, Transient, classify
from freelm.registry import resolve_models

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
OR_MODELS = "https://openrouter.ai/api/v1/models"
G_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"


# -- error taxonomy -------------------------------------------------------------


@pytest.mark.parametrize(
    "status,body,cls",
    [
        (520, "cloudflare: web server returned an unknown error", Transient),
        (524, "a timeout occurred", Transient),
        (501, "not implemented", Transient),
        (400, '{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.",'
              '"status":"INVALID_ARGUMENT","details":[{"reason":"API_KEY_INVALID"}]}}', AuthError),
        (400, '{"error":{"message":"User location is not supported for the API use."}}', AuthError),
        (403, '{"error":{"message":"Your input was flagged by moderation","code":403}}', BadRequest),
        (413, "Request too large for model `llama` on tokens per minute (TPM): Limit 6000", ModelNotFound),
        (400, "Please reduce the length of the messages or completion", ModelNotFound),
        (451, "unavailable for legal reasons", BadRequest),
        (418, "teapot", BadRequest),
    ],
)
def test_classify_never_returns_a_bare_abort(status, body, cls):
    assert type(classify(status, {}, body, "p")) is cls


def test_capability_404_does_not_bench_the_model():
    err = classify(404, {}, '{"error":{"message":"No endpoints found that support tool use."}}', "openrouter")
    assert isinstance(err, ModelNotFound) and err.gone is False
    gone = classify(404, {}, '{"error":{"message":"No endpoints found for x/y:free."}}', "openrouter")
    assert gone.gone is True


@respx.mock
def test_groq_413_tpm_fails_over_instead_of_raising():
    respx.post(GROQ_URL).mock(return_value=httpx.Response(413, text="Request too large ... tokens per minute (TPM)"))
    respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("g")))
    groq = Groq("gsk_test", discover=False, models=[ModelSpec("m", ("chat",))])
    with FreeLLM([groq, GoogleAIStudio("k")]) as llm:
        assert llm.chat("long prompt").provider == "google"
    assert groq.keys[0].disabled is False


@respx.mock
def test_moderation_403_does_not_disable_the_key():
    respx.post(OR_URL).mock(return_value=httpx.Response(403, text='{"error":{"message":"input was flagged"}}'))
    respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("g")))
    with FreeLLM([OpenRouter("k", discover=False), GoogleAIStudio("k2")]) as llm:
        assert llm.chat("hi").provider == "google"
        assert llm.providers[0].keys[0].disabled is False


@respx.mock
def test_402_naming_the_model_benches_model_not_key():
    body = '{"message":"Payment required to use gpt-oss-120b on the free tier"}'
    route = respx.post(CEREBRAS_URL).mock(
        side_effect=lambda req: httpx.Response(402, text=body)
        if json.loads(req.content)["model"] == "gpt-oss-120b"
        else httpx.Response(200, json=ok_payload("free-model"))
    )
    c = Cerebras("csk-test", discover=False, models=[ModelSpec("gpt-oss-120b", ("chat",)), ModelSpec("qwen", ("chat",))])
    with FreeLLM([c]) as llm:
        assert llm.chat("hi").text == "free-model"
        assert llm.chat("hi").text == "free-model"
    assert c.keys[0].disabled is False
    assert [json.loads(x.request.content)["model"] for x in route.calls] == ["gpt-oss-120b", "qwen", "qwen"]


@respx.mock
def test_402_account_wide_still_disables_key():
    respx.post(CEREBRAS_URL).mock(return_value=httpx.Response(402, text='{"message":"Payment required"}'))
    c = Cerebras("csk-test", discover=False)
    llm = FreeLLM([c])
    with pytest.raises(NoProvidersAvailable):
        llm.chat("hi")
    assert c.keys[0].disabled is True
    llm.close()


@respx.mock
def test_retry_after_zero_means_retry_now():
    err = classify(429, {"retry-after": "0"}, "slow down", "p")
    assert isinstance(err, RateLimited) and err.retry_after == 0.0
    respx.post(OR_URL).mock(side_effect=[httpx.Response(429, headers={"retry-after": "0"}),
                                         httpx.Response(200, json=ok_payload("ok"))])
    with FreeLLM([OpenRouter("k", discover=False)], wait=True, max_wait=5) as llm:
        k = llm.providers[0].keys[0]
        llm.chat("hi")
        assert k.cooldown_until <= time.monotonic()


def test_chat_rejects_stream_kwarg():
    with FreeLLM([OpenRouter("k", discover=False)]) as llm:
        with pytest.raises(ConfigError):
            llm.chat("hi", stream=True)


# -- secrets never in reprs ------------------------------------------------------------


@respx.mock
def test_reprs_never_contain_raw_keys():
    secret = "sk-or-v1-THIS-IS-A-VERY-SECRET-KEY-123456"
    respx.post(OR_URL).mock(return_value=httpx.Response(401, text="no"))
    llm = FreeLLM([OpenRouter(secret, discover=False)])
    with pytest.raises(NoProvidersAvailable) as ei:
        llm.chat("hi")
    dumps = [repr(ei.value.attempts), str(ei.value), repr(llm.providers[0].keys), repr(llm.health())]
    assert all("VERY-SECRET" not in d for d in dumps)
    llm.close()


# -- persistence ----------------------------------------------------------------------


def test_state_expired_daily_window_resets_counter(tmp_path):
    store = StateStore(str(tmp_path / "state.json"))
    p = OpenRouter("sk-or-abc", discover=False)
    k = p.keys[0]
    k.rpd_used = 50
    k.rpd_reset = 10.0  # window already over by the time we save
    store.save([p], now_mono=20.0)
    p2 = OpenRouter("sk-or-abc", discover=False)
    store.load_into([p2], 0.0)
    assert p2.keys[0].rpd_used == 0  # a new day, not another 24h of "exhausted"


def test_state_tolerates_bad_field_types(tmp_path):
    f = tmp_path / "state.json"
    p = OpenRouter("sk-or-abc", discover=False)
    from freelm._state import _key_id

    f.write_text(json.dumps({_key_id("openrouter", "sk-or-abc"): {"rpd_used": "n/a", "rpd_reset_wall": "x",
                                                                   "cooldown_until_wall": None, "disabled": False}}))
    StateStore(str(f)).load_into([p], 0.0)  # must not raise
    assert p.keys[0].rpd_used == 0


def test_state_disabled_flag_expires(tmp_path):
    from freelm._state import _key_id

    f = tmp_path / "state.json"
    old = time.time() - DISABLED_TTL - 60
    f.write_text(json.dumps({_key_id("openrouter", "k"): {"disabled": True, "disabled_since_wall": old}}))
    p = OpenRouter("k", discover=False)
    StateStore(str(f)).load_into([p], 0.0)
    assert p.keys[0].disabled is False  # a day later the key gets another chance

    f.write_text(json.dumps({_key_id("openrouter", "k"): {"disabled": True, "disabled_since_wall": time.time()}}))
    p = OpenRouter("k", discover=False)
    StateStore(str(f)).load_into([p], 0.0)
    assert p.keys[0].disabled is True


def test_state_save_leaves_no_temp_files(tmp_path):
    store = StateStore(str(tmp_path / "state.json"))
    p = OpenRouter("k", discover=False)
    for _ in range(5):
        store.save([p], 0.0)
    assert sorted(x.name for x in tmp_path.iterdir()) == ["state.json"]


@respx.mock
def test_reset_keys_gives_a_clean_slate():
    respx.post(OR_URL).mock(return_value=httpx.Response(401, text="bad"))
    llm = FreeLLM([OpenRouter("k", discover=False)])
    with pytest.raises(NoProvidersAvailable):
        llm.chat("hi")
    assert llm.providers[0].keys[0].disabled
    llm.reset_keys()
    assert not llm.providers[0].keys[0].disabled
    assert llm.health()[0]["ready"] is True
    llm.close()


# -- discovery & model lists ------------------------------------------------------------


@pytest.mark.parametrize(
    "mid,tags",
    [
        ("gemini-2.5-flash", []),
        ("minimax-m2", []),
        ("gpt-4o-mini", ["small", "fast"]),
        ("gemini-2.5-flash-lite", ["small", "fast"]),
        ("nemotron-3-super-120b-a12b", ["large"]),
        ("llama-3.1-405b-instruct", ["large"]),
        ("llama-3.1-8b-instant", ["small", "fast"]),
    ],
)
def test_size_tags_match_whole_tokens(mid, tags):
    assert _size_tags(mid) == tags


def test_to_specs_reads_groq_and_mistral_metadata_and_strips_google_prefix():
    specs = to_specs(
        [
            {"id": "llama-3.3-70b-versatile", "context_window": 131072},
            {"id": "mistral-small-latest", "max_context_length": 32000,
             "capabilities": {"completion_chat": True, "function_calling": True, "vision": True}},
            {"id": "mistral-embed", "capabilities": {"completion_chat": False}},
            {"id": "models/gemini-3.1-flash-lite"},
            {"id": "nvidia/nemotron-3.5-content-safety"},
        ],
        free_only=False,
    )
    by = {s.id: s for s in specs}
    assert set(by) == {"llama-3.3-70b-versatile", "mistral-small-latest", "gemini-3.1-flash-lite"}
    assert by["llama-3.3-70b-versatile"].ctx == 131072
    assert {"tools", "vision"} <= set(by["mistral-small-latest"].tags)
    assert by["mistral-small-latest"].ctx == 32000


def test_capability_aliases_do_not_fall_back_to_incapable_models():
    models = [ModelSpec("plain", ("chat",))]
    assert resolve_models(models, "chat:tools") == []
    assert resolve_models(models, "vision") == []
    assert resolve_models(models, "reasoning") == ["plain"]  # soft preference
    assert resolve_models(models, "chat:large") == ["plain"]  # soft preference


@respx.mock
def test_tools_request_skips_provider_without_tool_models():
    g = respx.post(G_URL).mock(return_value=httpx.Response(200, json=ok_payload("g")))
    orr = respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("or")))
    plain = OpenRouter("k", models=[ModelSpec("plain:free", ("chat",))])
    with FreeLLM([plain, GoogleAIStudio("k2")]) as llm:
        assert llm.chat("hi", model="chat:tools", tools=[{"type": "function"}]).provider == "google"
    assert orr.call_count == 0 and g.call_count == 1


def test_explicit_models_list_is_not_overwritten_by_discovery():
    p = OpenRouter("k", models=[ModelSpec("mine:free", ("chat",))])
    assert p.discover is False
    assert Groq("gsk", models=[ModelSpec("m", ("chat",))]).discover is False
    assert OpenRouter("k").discover is True


@respx.mock
def test_discovery_skips_a_dead_first_key():
    def models(req):
        if req.headers["authorization"] == "Bearer dead":
            return httpx.Response(401)
        return httpx.Response(200, json={"data": [{"id": "fresh/model", "context_window": 1000}]})

    route = respx.get("https://api.groq.com/openai/v1/models").mock(side_effect=models)
    respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=ok_payload("ok")))
    with FreeLLM([Groq(["dead", "alive"])]) as llm:
        llm.chat("hi")
        assert [m.id for m in llm.providers[0].models] == ["fresh/model"]
    assert route.call_count == 2


@respx.mock
def test_refresh_models_bypasses_disk_cache():
    route = respx.get(OR_MODELS).mock(return_value=httpx.Response(200, json={"data": [{"id": "a/b:free"}]}))
    respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("ok")))
    with FreeLLM([OpenRouter("k")]) as llm:
        llm.chat("hi")
        llm.refresh_models()
        llm.chat("hi")
    assert route.call_count == 2


def test_cache_with_wrong_shape_is_ignored(tmp_path, monkeypatch):
    from freelm import _cache

    monkeypatch.setenv("FREELM_CACHE_DIR", str(tmp_path))
    (tmp_path / "models-openrouter.json").write_text(json.dumps(["not", "a", "dict"]))
    assert _cache.load("openrouter") is None
    (tmp_path / "models-openrouter.json").write_text(json.dumps({"data": "nope", "expires_at": 9e12}))
    assert _cache.load("openrouter") is None


def test_async_concurrent_first_calls_discover_once():
    @respx.mock
    async def run():
        route = respx.get(OR_MODELS).mock(return_value=httpx.Response(200, json={"data": [{"id": "a/b:free"}]}))
        respx.post(OR_URL).mock(return_value=httpx.Response(200, json=ok_payload("ok")))
        async with AsyncFreeLLM([OpenRouter("k")]) as llm:
            await asyncio.gather(*(llm.chat("hi") for _ in range(10)))
        return route.call_count

    assert asyncio.run(run()) == 1



def test_discovery_skips_audio_generators_and_partially_priced_entries():
    specs = to_specs(
        [
            {"id": "google/lyria-3-pro-preview", "pricing": {"prompt": "0", "completion": "0"},
             "architecture": {"output_modalities": ["text", "audio"]}},
            {"id": "vendor/img-fee", "pricing": {"prompt": "0", "completion": "0", "image": "0.002"}},
            {"id": "inclusionai/ling-3.1-flash", "pricing": {"prompt": "0", "completion": "0"},
             "architecture": {"output_modalities": ["text"]}},
            {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
        ],
        free_only=True,
    )
    assert [s.id for s in specs] == ["inclusionai/ling-3.1-flash"]
