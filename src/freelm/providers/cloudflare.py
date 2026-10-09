"""Cloudflare Workers AI (https://developers.cloudflare.com/workers-ai/) via its
OpenAI-compatible endpoint. Needs an API token with Workers AI permission **and**
the account id (both on the Cloudflare dashboard).

Every account gets 10,000 Neurons/day free (a few hundred chats; checked
2026-10). On the Free plan requests fail past that — freelm then rests the key
and re-checks hourly; on the Workers Paid plan the overage is billed, so keep a
Free-plan account for freelm. Some newer models (Kimi K2.6, DeepSeek V4, GLM-5)
need Workers Paid and aren't listed. There is no OpenAI-style model list, so the
models below are curated. One provider per account: add a second as
``CloudflareWorkersAI(token, account_id=..., name="cloudflare-2")``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ..errors import ConfigError
from ..registry import ModelSpec
from .base import Provider

_ACCOUNT_ID = re.compile(r"[0-9a-f]{32}")


class CloudflareWorkersAI(Provider):
    name = "cloudflare"
    base_url = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"

    # text generation: 300 requests/minute (checked 2026-10); the daily Neuron
    # allowance is the real limit
    TIERS: Dict[str, Dict[str, Any]] = {"free": {"rpm": 300, "rpd": None}}

    # Free-plan models on 2026-10-09 (workers-ai/models), non-thinking first.
    DEFAULT_MODELS = [
        ModelSpec("@cf/meta/llama-4-scout-17b-16e-instruct", ("chat", "tools", "vision"), ctx=131000),
        ModelSpec("@cf/mistralai/mistral-small-3.1-24b-instruct", ("chat", "tools"), ctx=128000),
        ModelSpec("@cf/openai/gpt-oss-120b", ("chat", "large", "tools", "reasoning"), ctx=128000),
        ModelSpec("@cf/qwen/qwen3.8-27b", ("chat", "tools", "vision", "reasoning"), ctx=262144),
        ModelSpec("@cf/google/gemma-4-26b-a4b-it", ("chat", "tools", "vision", "reasoning"), ctx=256000),
        ModelSpec("@cf/zai-org/glm-4.7-flash", ("chat", "fast", "tools", "reasoning"), ctx=131072),
        ModelSpec("@cf/meta/llama-3.3-70b-instruct-fp8-fast", ("chat", "large"), ctx=24000),
        ModelSpec("@cf/openai/gpt-oss-20b", ("chat", "small", "fast", "tools", "reasoning"), ctx=128000),
        ModelSpec("@cf/meta/llama-3.1-8b-instruct-fp8", ("chat", "small", "fast"), ctx=32000),
    ]

    # Many models (Llama, gpt-oss, Mistral, Granite) default to 256 output
    # tokens, silently cutting answers short — send this unless the caller sets one.
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, keys, *, account_id: Optional[str] = None, **kw: Any) -> None:
        account_id = (account_id or "").strip().lower() or None
        if account_id is not None and not _ACCOUNT_ID.fullmatch(account_id):
            raise ConfigError("cloudflare: account_id must be the 32-character hex id from your Cloudflare dashboard")
        if not kw.get("base_url"):
            if account_id is None:
                raise ConfigError(
                    "cloudflare: account_id is required — set CLOUDFLARE_ACCOUNT_ID "
                    "(the 32-character id on your Cloudflare dashboard)"
                )
            kw["base_url"] = self.base_url.format(account_id=account_id)
        kw.setdefault("discover", False)  # no OpenAI-style model list
        super().__init__(keys, **kw)
        self.account_id = account_id

    def adapt_payload(self, body: Dict[str, Any]) -> Dict[str, Any]:
        if body.get("max_tokens") is None and body.get("max_completion_tokens") is None:
            body = {**body, "max_tokens": self.DEFAULT_MAX_TOKENS}
        return body

    def rate_limit_scope(self, body: str) -> str:
        # 3040 "Capacity temporarily exceeded" is one model; the daily Neuron
        # allowance (4006/3036) and the per-minute limit are the account's
        return "model" if "capacity" in (body or "").lower() else "key"
