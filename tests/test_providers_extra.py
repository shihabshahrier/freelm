from freelm import Cerebras, Groq, Mistral, providers_from_env


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

    for var in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY",
                "CEREBRAS_API_KEY", "MISTRAL_API_KEY", "KILO_API_KEY", "FREELM_KEYLESS"):
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
