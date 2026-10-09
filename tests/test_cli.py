import httpx
import pytest
import respx

from conftest import ok_payload
from freelm._cli import main
from freelm.config import PROVIDER_ENV

OR_CHAT = "https://openrouter.ai/api/v1/chat/completions"
OR_MODELS = "https://openrouter.ai/api/v1/models"

# every variable from_env() reads, so a developer's real env can't leak into a test
_ALL_KEY_VARS = [v for s in PROVIDER_ENV for v in (*s.key_vars, *(n for _, names in s.option_vars for n in names))]


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("FREELM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FREELM_KEYLESS", "0")  # tests opt in to keyless explicitly
    for var in _ALL_KEY_VARS:
        monkeypatch.delenv(var, raising=False)


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as ei:
        main(["--version"])
    assert ei.value.code == 0
    assert "freelm" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "chat" in capsys.readouterr().out


def test_no_keys_is_clean_config_error(capsys):
    assert main(["health"]) == 2
    assert "config error" in capsys.readouterr().err


@respx.mock
def test_chat_prints_reply(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    respx.get(OR_MODELS).mock(return_value=httpx.Response(500))  # discovery falls back
    respx.post(OR_CHAT).mock(return_value=httpx.Response(200, json=ok_payload("pong")))
    assert main(["chat", "ping"]) == 0
    out = capsys.readouterr()
    assert "pong" in out.out
    assert "openrouter" in out.err  # provider/model note goes to stderr


@respx.mock
def test_models_lists_fallback_catalog(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    respx.get(OR_MODELS).mock(return_value=httpx.Response(500))
    assert main(["models", "--provider", "openrouter"]) == 0
    out = capsys.readouterr().out
    assert "openrouter:" in out
    assert ":free" in out


@respx.mock
def test_health_prints_rows(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    respx.get(OR_MODELS).mock(return_value=httpx.Response(500))
    assert main(["health"]) == 0
    assert "openrouter" in capsys.readouterr().out


G_CHAT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def test_doctor_without_keys_prints_signup_links(capsys):
    assert main(["doctor"]) == 2
    out = capsys.readouterr().out
    assert "GEMINI_API_KEY" in out and "https://aistudio.google.com/apikey" in out


@respx.mock
def test_doctor_reports_each_key_with_a_fix(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-dead-key-123456")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-good-key-123456")
    respx.get(OR_MODELS).mock(return_value=httpx.Response(500))
    respx.post(OR_CHAT).mock(return_value=httpx.Response(401, json={"error": {"message": "User not found.", "code": 401}}))
    respx.post(G_CHAT).mock(return_value=httpx.Response(200, json=ok_payload("ok", model="gemini-2.5-flash-lite")))
    assert main(["doctor"]) == 0  # at least one key works
    out = capsys.readouterr().out
    assert "FAIL" in out and "User not found." in out and "https://openrouter.ai/keys" in out
    assert "OK" in out and "gemini-2.5-flash-lite" in out
    assert "1 of 2 key(s) working" in out
    assert "dead-key" not in out  # keys are masked


@respx.mock
def test_doctor_json_and_all_failing_exit_code(monkeypatch, capsys):
    import json

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-dead-key-123456")
    respx.get(OR_MODELS).mock(return_value=httpx.Response(500))
    respx.post(OR_CHAT).mock(return_value=httpx.Response(402, text="Payment required"))
    assert main(["doctor", "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["keys"][0]["status"] == "FAIL" and data["keys"][0]["works"] is False
    assert any(m["provider"] == "groq" for m in data["missing"])


KILO_CHAT = "https://api.kilo.ai/api/gateway/chat/completions"
OVH_CHAT = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions"


@respx.mock
def test_chat_without_keys_uses_keyless_endpoints_with_a_notice(monkeypatch, capsys):
    monkeypatch.delenv("FREELM_KEYLESS")  # CLI default: auto
    respx.get(url__regex=r".*/models$").mock(return_value=httpx.Response(500))
    route = respx.post(KILO_CHAT).mock(return_value=httpx.Response(200, json=ok_payload("hi from kilo")))
    respx.post(OVH_CHAT).mock(return_value=httpx.Response(429, text="limit"))
    assert main(["chat", "ping"]) == 0
    out = capsys.readouterr()
    assert "hi from kilo" in out.out
    assert "keyless public endpoints" in out.err
    assert "authorization" not in {k.lower() for k in route.calls[0].request.headers}


@respx.mock
def test_doctor_without_keys_checks_keyless(monkeypatch, capsys):
    monkeypatch.delenv("FREELM_KEYLESS")
    respx.get(url__regex=r".*/models$").mock(return_value=httpx.Response(500))
    respx.post(KILO_CHAT).mock(return_value=httpx.Response(200, json=ok_payload("ok", model="kilo-auto/free")))
    respx.post(OVH_CHAT).mock(return_value=httpx.Response(429, headers={"retry-after": "30"}, text="limit"))
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "GEMINI_API_KEY" in out  # still points to real free keys
    assert "keyless endpoints" in out and "(keyless)" in out
    assert "no keys configured, 2 keyless endpoint(s) up — ready: kilo, ovh" in out  # ovh: up but throttled


@respx.mock
def test_doctor_tests_keys_live_even_with_persisted_state(monkeypatch, tmp_path, capsys):
    import json as _json

    from freelm._state import _key_id

    monkeypatch.setenv("FREELM_PERSIST", "1")
    monkeypatch.setenv("FREELM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fixed-key-123456")
    (tmp_path / "state.json").write_text(_json.dumps(
        {_key_id("openrouter", "sk-or-fixed-key-123456"): {"disabled": True, "disabled_since_wall": 9e12}}))
    respx.get(OR_MODELS).mock(return_value=httpx.Response(500))
    route = respx.post(OR_CHAT).mock(return_value=httpx.Response(200, json=ok_payload("ok")))
    assert main(["doctor"]) == 0
    assert route.call_count == 1  # the saved "disabled" flag didn't short-circuit the check
    assert "OK" in capsys.readouterr().out


def test_doctor_explains_a_cloudflare_token_without_account_id(monkeypatch, capsys):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token-abcdefghijkl")
    assert main(["doctor"]) == 1
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.strip().startswith("cloudflare"))
    assert "FAIL" in line and "CLOUDFLARE_ACCOUNT_ID" in line
    assert "cf-token-abcdefghijkl" not in out  # masked


def test_doctor_lists_every_variable_a_provider_needs(capsys):
    assert main(["doctor"]) == 2  # nothing configured
    assert "export CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=..." in capsys.readouterr().out


@respx.mock
def test_doctor_shows_cloudflare_error_messages(monkeypatch, capsys):
    acct = "0123456789abcdef0123456789abcdef"
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token-abcdefghijkl")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", acct)
    respx.post(f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"result": None, "success": False, "messages": [],
                                               "errors": [{"code": 10000, "message": "Authentication error"}]}))
    assert main(["doctor"]) == 1
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.strip().startswith("cloudflare"))
    assert "key rejected (401): Authentication error" in line
