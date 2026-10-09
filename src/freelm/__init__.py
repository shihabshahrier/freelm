"""freelm — one always-up LLM client over free-tier providers.

Quick start::

    import freelm
    llm = freelm.FreeLLM.from_env()
    print(llm.text("Explain black holes in one sentence."))

Explicit config::

    from freelm import FreeLLM, OpenRouter, GoogleAIStudio, NIM
    llm = FreeLLM(
        providers=[OpenRouter("sk-or-..."), GoogleAIStudio("AIza..."), NIM("nvapi-...")],
        strategy="quota_aware",
    )
"""
from __future__ import annotations

from ._types import ChatRequest, ChatResponse, Choice, Event, Message, Usage
from ._version import __version__
from .client import AsyncFreeLLM, FreeLLM
from .config import providers_from_env
from .discovery import list_free_models
from .errors import (
    AuthError,
    BadRequest,
    ConfigError,
    FreeLLMError,
    ModelNotFound,
    NoProvidersAvailable,
    ProviderError,
    QuotaExhausted,
    RateLimited,
    Transient,
)
from .providers import NIM, Cerebras, Gemini, GoogleAIStudio, Groq, Kilo, Mistral, OpenRouter, OVHcloud, Provider
from .registry import ModelSpec


def serve(*args, **kwargs):
    """Run the local OpenAI-compatible endpoint (``freelm serve``); see :mod:`freelm.server`."""
    from .server import serve as _serve

    return _serve(*args, **kwargs)


__all__ = [
    "FreeLLM",
    "AsyncFreeLLM",
    "Provider",
    "OpenRouter",
    "GoogleAIStudio",
    "Gemini",
    "NIM",
    "Groq",
    "Cerebras",
    "Mistral",
    "Kilo",
    "OVHcloud",
    "ModelSpec",
    "Message",
    "ChatRequest",
    "ChatResponse",
    "Choice",
    "Usage",
    "Event",
    "providers_from_env",
    "list_free_models",
    "serve",
    "FreeLLMError",
    "ConfigError",
    "ProviderError",
    "AuthError",
    "BadRequest",
    "QuotaExhausted",
    "RateLimited",
    "Transient",
    "ModelNotFound",
    "NoProvidersAvailable",
    "__version__",
]
