from freelm import Cerebras, Groq, Mistral, providers_from_env


def test_new_providers_construct():
    for P, host in [(Groq, "groq.com"), (Cerebras, "cerebras.ai"), (Mistral, "mistral.ai")]:
        p = P("key")
        assert host in p.url
        assert p.url.endswith("/chat/completions")
        assert p.resolve_models("auto")  # non-empty default models
        assert p.headers("key")["Authorization"] == "Bearer key"


def test_from_env_includes_new_providers(monkeypatch):
    for var in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gk")
    monkeypatch.setenv("CEREBRAS_API_KEY", "ck")
    monkeypatch.setenv("MISTRAL_API_KEY", "mk")
    names = {p.name for p in providers_from_env()}
    assert {"groq", "cerebras", "mistral"} <= names
