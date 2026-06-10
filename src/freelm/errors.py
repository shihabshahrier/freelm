"""Exception hierarchy and HTTP-status -> error classification."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class FreeLLMError(Exception):
    """Base class for all freelm errors."""


class ConfigError(FreeLLMError):
    """Bad or missing configuration (no keys, unknown provider, ...)."""


class ProviderError(FreeLLMError):
    """A provider returned an error response."""

    def __init__(
        self,
        provider: str,
        status: int,
        message: str = "",
        *,
        retryable: bool = False,
        retry_after: Optional[float] = None,
        model_missing: bool = False,
    ) -> None:
        super().__init__(f"[{provider}] {status} {message}".strip())
        self.provider = provider
        self.status = status
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after
        self.model_missing = model_missing


class AuthError(ProviderError):
    """401/403 — key is invalid or lacks access. Key gets disabled."""

    def __init__(self, provider: str, status: int, message: str = "") -> None:
        super().__init__(provider, status, message, retryable=False)


class RateLimited(ProviderError):
    """429 — rotate to another key/provider, cool this key down.

    ``scope`` is ``"key"`` (account/key-wide, default) or ``"model"`` (only this
    model is throttled upstream — try a different model on the same key first).
    """

    def __init__(
        self,
        provider: str,
        status: int,
        message: str = "",
        retry_after: Optional[float] = None,
        scope: str = "key",
    ) -> None:
        super().__init__(provider, status, message, retryable=True, retry_after=retry_after)
        self.scope = scope


class Transient(ProviderError):
    """Timeout / 5xx — back off and retry elsewhere."""

    def __init__(self, provider: str, status: int, message: str = "", retry_after: Optional[float] = None) -> None:
        super().__init__(provider, status, message, retryable=True, retry_after=retry_after)


class ModelNotFound(ProviderError):
    """The requested model is unknown to this provider — try another model/provider."""

    def __init__(self, provider: str, status: int, message: str = "") -> None:
        super().__init__(provider, status, message, retryable=False, model_missing=True)


class QuotaExhausted(ProviderError):
    """402 — the account is out of credits/quota (e.g. OpenRouter below the free
    threshold). The key is disabled for this process and we fail over, like an
    auth error — it won't recover without human action."""

    def __init__(self, provider: str, status: int, message: str = "") -> None:
        super().__init__(provider, status, message, retryable=False)


class NoProvidersAvailable(FreeLLMError):
    """Every candidate provider/key was exhausted or unavailable."""

    def __init__(self, attempts: List[Tuple[Any, Exception]]) -> None:
        self.attempts = attempts
        detail = "; ".join(
            "{}/{}:{}".format(c.provider.name, c.key.masked(), type(e).__name__) for c, e in attempts[:8]
        )
        super().__init__(
            "all providers/keys exhausted after {} attempt(s): {}".format(len(attempts), detail or "none ready")
        )


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:  # HTTP-date form
        import datetime
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(value)
        now = datetime.datetime.now(dt.tzinfo)
        return max(0.0, (dt - now).total_seconds())
    except Exception:
        return None


_TRANSIENT_STATUS = {408, 409, 425, 500, 502, 503, 504, 529}


def classify(status: int, headers: Optional[Dict[str, str]], body: str, provider: str) -> ProviderError:
    """Map an HTTP error response to the right exception."""
    retry_after = parse_retry_after((headers or {}).get("retry-after"))
    msg = (body or "")[:300]
    low = (body or "").lower()
    if status in (401, 403):
        return AuthError(provider, status, msg)
    if status == 402:
        return QuotaExhausted(provider, status, msg)
    if status == 429:
        return RateLimited(provider, status, msg, retry_after=retry_after)
    if status in _TRANSIENT_STATUS:
        return Transient(provider, status, msg, retry_after=retry_after)
    if status == 404 or (status in (400, 422) and "model" in low):
        return ModelNotFound(provider, status, msg)
    return ProviderError(provider, status, msg, retryable=False)
