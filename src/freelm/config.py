"""Build providers from environment variables.

Recognised vars (comma-separate to supply multiple keys per provider):

  OpenRouter : OPENROUTER_API_KEY   | FREELM_OPENROUTER_KEYS   (+ FREELM_OPENROUTER_TIER)
  Google     : GEMINI_API_KEY / GOOGLE_API_KEY / GOOGLE_AI_STUDIO_KEY | FREELM_GOOGLE_KEYS (+ FREELM_GOOGLE_TIER)
  NVIDIA NIM : NVIDIA_API_KEY / NIM_API_KEY | FREELM_NIM_KEYS         (+ FREELM_NIM_TIER)
  Groq       : GROQ_API_KEY     | FREELM_GROQ_KEYS     (+ FREELM_GROQ_TIER)
  Cerebras   : CEREBRAS_API_KEY | FREELM_CEREBRAS_KEYS (+ FREELM_CEREBRAS_TIER)
  Mistral    : MISTRAL_API_KEY  | FREELM_MISTRAL_KEYS  (+ FREELM_MISTRAL_TIER)
  Z.ai       : ZAI_API_KEY      | FREELM_ZAI_KEYS      (+ FREELM_ZAI_TIER)
  Cohere     : COHERE_API_KEY / CO_API_KEY | FREELM_COHERE_KEYS (+ FREELM_COHERE_TIER)
  Cloudflare : CLOUDFLARE_API_TOKEN | FREELM_CLOUDFLARE_KEYS, plus CLOUDFLARE_ACCOUNT_ID (required)
  Kilo       : KILO_API_KEY     | FREELM_KILO_KEYS     (optional: works keyless too)

Keyless public endpoints (Kilo Gateway, OVHcloud) need no signup at all. They
are added only when asked: ``keyless=True`` / ``FREELM_KEYLESS=1`` (always, as
last-resort fallbacks) or ``"auto"`` (only when no keys are configured — what
the CLI uses). The library never routes prompts to them silently.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Type, Union

from .errors import ConfigError
from .providers import (
    NIM,
    ZAI,
    Cerebras,
    CloudflareWorkersAI,
    Cohere,
    GoogleAIStudio,
    Groq,
    Kilo,
    Mistral,
    OpenRouter,
    OVHcloud,
)
from .providers.base import Provider


@dataclass(frozen=True)
class ProviderEnv:
    """How one provider is configured from the environment."""

    name: str
    cls: Type[Provider]
    key_vars: Tuple[str, ...]  # first non-empty wins; comma-separated = several keys
    tier_var: str
    signup_url: str  # where to get a free key
    # extra constructor options from the env: ((kwarg, (VAR, ...)), ...), first non-empty VAR wins
    option_vars: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()


# Order = default provider order for `from_env()`.
# NB: Groq (gsk_...) is the free provider here; xAI Grok (xai-...) is a
# different, paid service and is intentionally not supported.
PROVIDER_ENV: Tuple[ProviderEnv, ...] = (
    ProviderEnv("openrouter", OpenRouter, ("OPENROUTER_API_KEY", "FREELM_OPENROUTER_KEYS"),
                "FREELM_OPENROUTER_TIER", "https://openrouter.ai/keys"),
    ProviderEnv("google", GoogleAIStudio, ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_KEY",
                                           "FREELM_GOOGLE_KEYS"),
                "FREELM_GOOGLE_TIER", "https://aistudio.google.com/apikey"),
    ProviderEnv("nim", NIM, ("NVIDIA_API_KEY", "NIM_API_KEY", "FREELM_NIM_KEYS"),
                "FREELM_NIM_TIER", "https://build.nvidia.com/settings/api-keys"),
    ProviderEnv("groq", Groq, ("GROQ_API_KEY", "FREELM_GROQ_KEYS"),
                "FREELM_GROQ_TIER", "https://console.groq.com/keys"),
    ProviderEnv("cerebras", Cerebras, ("CEREBRAS_API_KEY", "FREELM_CEREBRAS_KEYS"),
                "FREELM_CEREBRAS_TIER", "https://cloud.cerebras.ai"),
    ProviderEnv("mistral", Mistral, ("MISTRAL_API_KEY", "FREELM_MISTRAL_KEYS"),
                "FREELM_MISTRAL_TIER", "https://console.mistral.ai/api-keys"),
    ProviderEnv("zai", ZAI, ("ZAI_API_KEY", "FREELM_ZAI_KEYS"),
                "FREELM_ZAI_TIER", "https://z.ai/manage-apikey/apikey-list"),
    ProviderEnv("cohere", Cohere, ("COHERE_API_KEY", "CO_API_KEY", "FREELM_COHERE_KEYS"),
                "FREELM_COHERE_TIER", "https://dashboard.cohere.com/api-keys"),
    ProviderEnv("cloudflare", CloudflareWorkersAI, ("CLOUDFLARE_API_TOKEN", "FREELM_CLOUDFLARE_KEYS"),
                "FREELM_CLOUDFLARE_TIER", "https://dash.cloudflare.com/profile/api-tokens",
                (("account_id", ("CLOUDFLARE_ACCOUNT_ID",)),)),
    ProviderEnv("kilo", Kilo, ("KILO_API_KEY", "FREELM_KILO_KEYS"),
                "FREELM_KILO_TIER", "https://app.kilo.ai"),
)

# No-signup endpoints, tried last (priority 100) when keyless mode is on.
KEYLESS: Tuple[Type[Provider], ...] = (Kilo, OVHcloud)
KEYLESS_NOTICE = (
    "freelm: no API keys found — using keyless public endpoints (Kilo Gateway, OVHcloud). "
    "Limits are low and free routes may log prompts; add a free key for more: `freelm doctor`."
)
KeylessArg = Union[bool, str, None]


def _split(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _first_env(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def env_keys(spec: ProviderEnv) -> List[str]:
    """The keys configured for one provider (empty list if none)."""
    return _split(_first_env(*spec.key_vars))


def env_vars(spec: ProviderEnv) -> List[str]:
    """The variables to set for one provider (key first), for setup hints."""
    return [spec.key_vars[0]] + [names[0] for _, names in spec.option_vars]


def build_provider(spec: ProviderEnv, keys: Union[str, Sequence[str]]) -> Provider:
    """``spec``'s provider for ``keys``, with its tier and options from the env.
    Raises ``ConfigError`` when a required option is missing (Cloudflare's
    account id)."""
    opts = {kw: v for kw, names in spec.option_vars for v in [_first_env(*names)] if v}
    return spec.cls(keys, tier=os.getenv(spec.tier_var, "free"), **opts)


def keyless_mode(value: KeylessArg = None) -> str:
    """``"always"`` | ``"auto"`` | ``"never"`` from an argument or ``FREELM_KEYLESS``
    (library default: never)."""
    if value is True:
        return "always"
    if value is False:
        return "never"
    v = str(value if value is not None else os.getenv("FREELM_KEYLESS", "")).strip().lower()
    if v in ("1", "true", "yes", "on", "always"):
        return "always"
    if v == "auto":
        return "auto"
    return "never"


def providers_from_env(keyless: KeylessArg = None) -> List[Provider]:
    """Providers for every key found in the environment, plus the keyless
    endpoints when ``keyless`` (or ``FREELM_KEYLESS``) asks for them."""
    provs: List[Provider] = []
    for spec in PROVIDER_ENV:
        keys = env_keys(spec)
        if keys:
            try:
                provs.append(build_provider(spec, keys))
            except ConfigError as e:  # e.g. a Cloudflare token without its account id
                print(f"freelm: {e}; skipping it.", file=sys.stderr)

    mode = keyless_mode(keyless)
    if mode == "always" or (mode == "auto" and not provs):
        have = {p.name for p in provs}
        keyed = bool(provs)
        provs += [cls(priority=100) for cls in KEYLESS if cls.name not in have]
        if not keyed:
            print(KEYLESS_NOTICE, file=sys.stderr)

    if not provs:
        raise ConfigError(
            "no provider keys found in environment. Set at least one of "
            + ", ".join(spec.key_vars[0] for spec in PROVIDER_ENV if spec.cls is not Kilo)
            + " (free keys: run `freelm doctor` for signup links), or set FREELM_KEYLESS=1 to use "
            "keyless public endpoints (Kilo Gateway, OVHcloud)."
        )
    return provs
