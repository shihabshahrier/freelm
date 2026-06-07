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

from ._types import ChatRequest, ChatResponse, Choice, Message, Usage
from .client import AsyncFreeLLM, FreeLLM
from .config import providers_from_env
from .discovery import list_free_models
from .errors import (
    AuthError,
    ConfigError,
    FreeLLMError,
    ModelNotFound,
    NoProvidersAvailable,
    ProviderError,
    RateLimited,
    Transient,
)
from .providers import Cerebras, Gemini, GoogleAIStudio, Groq, Mistral, NIM, OpenRouter, Provider
from .registry import ModelSpec

__version__ = "0.2.1"

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
    "ModelSpec",
    "Message",
    "ChatRequest",
    "ChatResponse",
    "Choice",
    "Usage",
    "providers_from_env",
    "list_free_models",
    "FreeLLMError",
    "ConfigError",
    "ProviderError",
    "AuthError",
    "RateLimited",
    "Transient",
    "ModelNotFound",
    "NoProvidersAvailable",
    "__version__",
]
