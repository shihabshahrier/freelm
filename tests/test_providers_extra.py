import json

import httpx
import pytest
import respx

from conftest import ok_payload
from freelm import Cerebras, Groq, Mistral, providers_from_env
from freelm.config import PROVIDER_ENV

# every variable from_env() reads, so a developer's real env can't leak into a test
ALL_ENV_VARS = [v for s in PROVIDER_ENV for v in (*s.key_vars, *(n for _, names in s.option_vars for n in names))]


def test_new_providers_construct():
    for P, host in [(Groq, "groq.com"), (Cerebras, "cerebras.ai"), (Mistral, "mistral.ai")]:
        p = P("key")
        assert host in p.url
        assert p.url.endswith("/chat/completions")
        assert p.resolve_models("auto")  # non-empty default models
        assert p.headers("key")["Authorization"] == "Bearer key"


def test_new_providers_have_discovery_enabled():
    # runtime /models discovery self-corrects their model IDs
    assert Groq("k").discover is True
    assert Cerebras("k").discover is True
    assert Mistral("k").discover is True


def test_from_env_includes_new_providers(monkeypatch):
    for var in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gk")
    monkeypatch.setenv("CEREBRAS_API_KEY", "ck")
    monkeypatch.setenv("MISTRAL_API_KEY", "mk")
    names = {p.name for p in providers_from_env()}
    assert {"groq", "cerebras", "mistral"} <= names



# -- keyless providers + keyless mode ------------------------------------------------


def test_keyless_providers_send_no_credentials():
    from freelm import Kilo, OVHcloud

    k = Kilo()
    assert k.headers(k.keys[0].key).get("Authorization") is None
    assert k.keys[0].masked() == "(keyless)"
    assert Kilo("my-kilo-key").headers("my-kilo-key")["Authorization"] == "Bearer my-kilo-key"
    o = OVHcloud(keys=["paid-key"])  # an OVH key would be pay-as-you-go: ignored
    assert o.keys[0].masked() == "(keyless)"
    assert o.rate_limit_scope("") == "model"


def test_kilo_is_free_only():
    import pytest

    from freelm import ConfigError, Kilo

    k = Kilo(discover=False)
    assert k.resolve_models("poolside/laguna-s-2.1:free") == ["poolside/laguna-s-2.1:free"]
    assert k.resolve_models("kilo-auto/free") == ["kilo-auto/free"]
    with pytest.raises(ConfigError):
        k.resolve_models("anthropic/claude-sonnet-4.5")


def test_keyless_modes(monkeypatch):
    import pytest

    from freelm import ConfigError, providers_from_env

    for var in (*ALL_ENV_VARS, "FREELM_KEYLESS"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfigError, match="FREELM_KEYLESS=1"):
        providers_from_env()  # the library never goes keyless on its own
    assert [p.name for p in providers_from_env(keyless="auto")] == ["kilo", "ovh"]
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-x")
    assert [p.name for p in providers_from_env(keyless="auto")] == ["google"]
    provs = providers_from_env(keyless=True)
    assert [p.name for p in provs] == ["google", "kilo", "ovh"]
    assert provs[1].priority == 100  # keyless are last resorts
    monkeypatch.setenv("FREELM_KEYLESS", "1")
    assert [p.name for p in providers_from_env()] == ["google", "kilo", "ovh"]


# -- Z.ai, Cohere, Cloudflare Workers AI ----------------------------------------------

CF_ACCOUNT = "0123456789abcdef0123456789abcdef"
CF_CHAT = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT}/ai/v1/chat/completions"
ZAI_CHAT = "https://api.z.ai/api/paas/v4/chat/completions"


def _cf(**kw):
    from freelm import CloudflareWorkersAI

    return CloudflareWorkersAI("cf-token-0123456789", account_id=CF_ACCOUNT, **kw)


def test_more_free_providers_construct():
    from freelm import ZAI, CloudflareWorkersAI, Cohere

    for p, url in [
        (ZAI("zk"), ZAI_CHAT),
        (Cohere("ck"), "https://api.cohere.ai/compatibility/v1/chat/completions"),
        (CloudflareWorkersAI("cft", account_id=CF_ACCOUNT), CF_CHAT),
    ]:
        assert p.url == url
        assert p.headers(p.keys[0].key)["Authorization"] == f"Bearer {p.keys[0].key}"
        assert p.resolve_models("auto")
        assert p.discover is False  # no usable /models list: curated defaults
    # limits are per model on Z.ai and Cohere; Cloudflare's daily allowance is the account's
    assert ZAI("k").rate_limit_scope("") == "model" and Cohere("k").rate_limit_scope("") == "model"
    assert _cf().rate_limit_scope('{"errors":[{"code":3040,"message":"Capacity temporarily exceeded"}]}') == "model"
    assert _cf().rate_limit_scope('{"errors":[{"code":4006,"message":"daily free allocation"}]}') == "key"


def test_zai_only_offers_its_free_flash_models():
    from freelm import ZAI, ConfigError

    z = ZAI("k")
    assert z.resolve_models("vision") == ["glm-4.6v-flash"]
    assert z.resolve_models("glm-4.5-flash") == ["glm-4.5-flash"]
    with pytest.raises(ConfigError, match="not a free model"):
        z.resolve_models("glm-4.6")  # billed against the account balance


def test_cloudflare_needs_a_valid_account_id():
    from freelm import CloudflareWorkersAI, ConfigError

    with pytest.raises(ConfigError, match="CLOUDFLARE_ACCOUNT_ID"):
        CloudflareWorkersAI("tok")
    with pytest.raises(ConfigError, match="32-character"):
        CloudflareWorkersAI("tok", account_id="my-account")
    assert CloudflareWorkersAI("tok", account_id=CF_ACCOUNT.upper()).account_id == CF_ACCOUNT
    # an explicit base_url (e.g. an AI Gateway route) needs no account id
    assert CloudflareWorkersAI("tok", base_url="https://gw.example/v1").url == "https://gw.example/v1/chat/completions"


def test_from_env_builds_the_new_providers(monkeypatch, capsys):
    for var in (*ALL_ENV_VARS, "FREELM_KEYLESS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ZAI_API_KEY", "zk")
    monkeypatch.setenv("CO_API_KEY", "ck")  # the Cohere SDK's variable works too
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cft")
    # a token without its account id is skipped with a notice, not a crash
    assert [p.name for p in providers_from_env()] == ["zai", "cohere"]
    assert "CLOUDFLARE_ACCOUNT_ID" in capsys.readouterr().err
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", CF_ACCOUNT)
    cf = providers_from_env()[-1]
    assert cf.name == "cloudflare" and cf.url == CF_CHAT


@respx.mock
def test_cloudflare_requests_get_room_to_answer():
    from freelm import FreeLLM

    model = "@cf/meta/llama-4-scout-17b-16e-instruct"
    route = respx.post(CF_CHAT).mock(return_value=httpx.Response(200, json=ok_payload("hi", model=model)))
    with FreeLLM([_cf()]) as llm:
        assert llm.chat("hello").text == "hi"
        llm.chat("hello", max_tokens=50)
    first, second = (json.loads(c.request.content) for c in route.calls)
    assert route.calls[0].request.headers["authorization"] == "Bearer cf-token-0123456789"
    assert first["model"] == model
    assert first["max_tokens"] == 4096  # Workers AI would stop at 256 tokens
    assert second["max_tokens"] == 50  # the caller's choice wins


@respx.mock
def test_numeric_tokens_become_text():
    from freelm import FreeLLM

    sse = (
        'data: {"choices":[{"index":0,"delta":{"content":"2 + 4 = "}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{"content":6}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    whole = ok_payload()
    whole["choices"][0]["message"]["content"] = 6

    def reply(req):
        if json.loads(req.content).get("stream"):
            return httpx.Response(200, text=sse)
        return httpx.Response(200, json=whole)

    respx.post(CF_CHAT).mock(side_effect=reply)
    with FreeLLM([_cf()]) as llm:
        assert "".join(llm.stream("2+4?")) == "2 + 4 = 6"  # the 6 used to vanish
        assert list(llm.stream_chunks("2+4?"))[1]["choices"][0]["delta"]["content"] == "6"
        assert llm.chat("2+4?").text == "6"


def test_quota_429s_rest_the_key_instead_of_retrying_every_minute():
    from freelm.errors import DAILY_QUOTA_RETRY, QuotaExhausted, RateLimited, classify

    cf_daily = json.dumps({"result": None, "success": False, "messages": [], "errors": [{"code": 4006, "message": (
        "you have used up your daily free allocation of 10,000 neurons, please upgrade to Cloudflare's "
        "Workers Paid plan if you would like to continue usage.")}]})
    e = classify(429, None, cf_daily, "cloudflare")
    assert isinstance(e, RateLimited) and e.retry_after == DAILY_QUOTA_RETRY
    or_daily = '{"error":{"message":"Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day","code":429}}'
    assert classify(429, None, or_daily, "openrouter").retry_after == DAILY_QUOTA_RETRY
    assert classify(429, {"retry-after": "7"}, or_daily, "openrouter").retry_after == 7  # a given wait wins
    per_minute = '{"id":"x","message":"You are using a Trial key, which is limited to 20 API calls / minute."}'
    assert classify(429, None, per_minute, "cohere").retry_after is None  # the usual short cooldown
    monthly = '{"id":"x","message":"You are using a Trial key, which is limited to 1000 API calls / month."}'
    assert isinstance(classify(429, None, monthly, "cohere"), QuotaExhausted)  # disables the key


def test_a_model_missing_from_the_plan_benches_the_model_not_the_key():
    from freelm.errors import ModelNotFound, classify

    paid_only = '{"errors":[{"code":5035,"message":"This model requires the Workers Paid plan."}],"success":false}'
    e = classify(403, None, paid_only, "cloudflare")
    assert isinstance(e, ModelNotFound) and e.gone and not e.retired
    e = classify(400, None, '{"errors":[{"code":5007,"message":"No such model @cf/x/y or task"}]}', "cloudflare")
    assert isinstance(e, ModelNotFound) and e.gone


@respx.mock
def test_cloudflare_daily_allowance_rests_the_key_for_an_hour():
    import time

    from freelm import FreeLLM, NoProvidersAvailable

    body = {"success": False, "errors": [{"code": 4006, "message": "you have used up your daily free allocation of 10,000 neurons"}]}
    route = respx.post(CF_CHAT).mock(return_value=httpx.Response(429, json=body))
    p = _cf()
    with FreeLLM([p]) as llm:
        with pytest.raises(NoProvidersAvailable):
            llm.chat("hi")
    assert route.call_count == 1  # not every model in turn: the account is out
    assert p.keys[0].cooldown_until - time.monotonic() > 3000


@respx.mock
def test_zai_429_benches_only_that_model():
    from freelm import ZAI, FreeLLM

    def reply(req):
        model = json.loads(req.content)["model"]
        if model == "glm-4.7-flash":
            return httpx.Response(429, json={"error": {"code": "1302", "message": "Rate limit reached for requests"}})
        return httpx.Response(200, json=ok_payload("ok", model=model))

    route = respx.post(ZAI_CHAT).mock(side_effect=reply)
    p = ZAI("zk")
    with FreeLLM([p]) as llm:
        assert llm.chat("a").model == "glm-4.5-flash"
        assert llm.chat("b").model == "glm-4.5-flash"  # the benched model isn't retried
    assert [json.loads(c.request.content)["model"] for c in route.calls] == [
        "glm-4.7-flash", "glm-4.5-flash", "glm-4.5-flash"]
    assert p.keys[0].cooldown_until == 0.0  # the key itself stays hot


def test_openrouter_sends_app_attribution_headers():
    from freelm import OpenRouter

    h = OpenRouter("k", discover=False).headers("k")
    assert h["HTTP-Referer"] == "https://github.com/shihabshahrier/freelm"
    assert h["X-OpenRouter-Title"] == "freelm" and h["X-Title"] == "freelm"  # current + legacy name
    assert h["X-OpenRouter-Categories"] == "programming-app,general-chat"  # at most 2 per request
    custom = OpenRouter("k", discover=False, extra_headers={"X-OpenRouter-Title": "my-app"}).headers("k")
    assert custom["X-OpenRouter-Title"] == "my-app"  # callers can attribute their own app

