import httpx
import pytest
import respx

from conftest import ok_payload

from freelm._cli import main

OR_CHAT = "https://openrouter.ai/api/v1/chat/completions"
OR_MODELS = "https://openrouter.ai/api/v1/models"

_ALL_KEY_VARS = (
    "OPENROUTER_API_KEY", "FREELM_OPENROUTER_KEYS",
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_KEY", "FREELM_GOOGLE_KEYS",
    "NVIDIA_API_KEY", "NIM_API_KEY", "FREELM_NIM_KEYS",
    "GROQ_API_KEY", "FREELM_GROQ_KEYS",
    "CEREBRAS_API_KEY", "FREELM_CEREBRAS_KEYS",
    "MISTRAL_API_KEY", "FREELM_MISTRAL_KEYS",
)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("FREELM_CACHE_DIR", str(tmp_path))
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
