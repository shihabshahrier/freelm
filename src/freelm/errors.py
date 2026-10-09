"""Exception hierarchy and HTTP-status -> error classification."""
from __future__ import annotations

import re
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

    ``scope`` is ``"key"`` (this key/account is throttled — cool it, default),
    ``"model"`` (this key's quota for this one model — bench the pair, other
    models stay usable) or ``"upstream"`` (the model is throttled upstream for
    everyone, e.g. OpenRouter — bench it on every key).
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
    """Timeout / 5xx — back off and retry elsewhere.

    ``scope`` is ``"key"`` (default: the provider/key is struggling — cool it)
    or ``"model"`` (only this model is overloaded — bench it, keep the key)."""

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


class ModelNotFound(ProviderError):
    """This model can't serve the request on this provider — try another model/provider.

    ``gone`` is True when the model id is unknown or retired (404/410, "no longer
    available", "decommissioned", ...): the provider remembers it and stops
    offering it for a while. False means only *this* request was rejected for
    the model (e.g. context too long), so the model stays in rotation.
    """

    def __init__(self, provider: str, status: int, message: str = "", gone: bool = False,
                 retired: bool = False) -> None:
        super().__init__(provider, status, message, retryable=False, model_missing=True)
        self.gone = gone
        # retired for everyone (410, "end of life", "decommissioned", ...) vs
        # merely unavailable to this key ("does not exist or you do not have access")
        self.retired = retired


class QuotaExhausted(ProviderError):
    """402 — the account is out of credits/quota (e.g. OpenRouter below the free
    threshold). The key is disabled for this process and we fail over, like an
    auth error — it won't recover without human action."""

    def __init__(self, provider: str, status: int, message: str = "") -> None:
        super().__init__(provider, status, message, retryable=False)


class BadRequest(ProviderError):
    """This provider rejected the request itself (an unrecognised 4xx — e.g. a
    parameter it doesn't support, or content its moderation flagged).

    Free tiers disagree about what they accept, so the call fails over to the
    next *provider*. Once two different providers reject the same request (or
    every reachable one has), it is treated as a caller bug and raised."""

    def __init__(self, provider: str, status: int, message: str = "") -> None:
        super().__init__(provider, status, message, retryable=False)


class NoProvidersAvailable(FreeLLMError):
    """Every candidate provider/key was exhausted or unavailable.

    ``attempts`` is ``[(candidate, exception), ...]`` for this call; ``status``
    holds one human-readable line per provider explaining why it's unusable
    (disabled key, cooling down, no usable model, ...)."""

    def __init__(self, attempts: List[Tuple[Any, Exception]], status: Optional[List[str]] = None) -> None:
        self.attempts = attempts
        self.status = list(status or [])
        detail = "; ".join(
            "{}/{}:{}{}".format(
                c.provider.name,
                c.model,
                type(e).__name__,
                "({})".format(e.status) if isinstance(e, ProviderError) and e.status else "",
            )
            for c, e in attempts[:8]
        )
        msg = "all providers/keys exhausted after {} attempt(s): {}".format(len(attempts), detail or "none ready")
        if self.status:
            msg += ". Provider status: " + " | ".join(self.status)
        if any(isinstance(e, AuthError) for _, e in attempts) or any("invalid/expired" in s for s in self.status):
            msg += ". Run `freelm doctor` to check your keys."
        super().__init__(msg)


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


# Retryable statuses besides the whole 5xx range (Cloudflare 52x, 501, ...).
_TRANSIENT_STATUS = {408, 409, 425}
# 404 Not Found / 410 Gone: the model id is unknown or retired on this provider
# (e.g. NVIDIA NIM answers 410 for end-of-life models).
_GONE_STATUS = {404, 410}
# A model retired for everyone (vs. one this key can't access).
_RETIRED_HINTS = ("no longer", "decommission", "deprecated", "end of life", "end-of-life", "retired")
# Phrases that mark a 400/422 "model" error as permanent for that model id,
# not just a problem with this particular request.
_GONE_HINTS = (
    "no longer", "decommission", "deprecated", "end of life", "end-of-life", "retired",
    "not found", "not_found", "does not exist", "unknown model", "invalid model",
    "model not available", "no such model",
)
# The model exists but lacks a capability this request needs (tools, images, ...).
_CAPABILITY_HINTS = ("support", "tool use", "not enabled")
# The request is too big for this model (another model/provider may take it):
# context window overflow, Groq's 413 "Request too large ... tokens per minute".
_TOO_BIG_HINTS = (
    "context", "too long", "too large", "reduce the length", "maximum length",
    "token limit", "tokens per minute",
)
# Bad-key errors some providers send as 400 (Google: API_KEY_INVALID).
_AUTH_HINTS = (
    "api key not valid", "api_key_invalid", "api key expired", "api_key_expired",
    "invalid api key", "invalid_api_key", "incorrect api key",
)
# A 403 about the *content* (OpenRouter moderation), not the key.
_MODERATION_HINTS = ("flagged", "moderation")
# A model this account's plan doesn't include (Cloudflare's Workers Paid-only
# models answer 403 on the Free plan) — the key itself is fine.
_PLAN_HINTS = ("workers paid", "paid plan")
# A 429 meaning the quota is spent for the month (Cohere trial keys) — no
# recovery soon, like a 402 — or for the day (Cloudflare's daily free Neurons,
# OpenRouter's free-models-per-day): when no retry time is given, look again in
# an hour rather than every minute (daily resets aren't reliably at 00:00 UTC).
_MONTHLY_HINTS = ("/ month", "per month", "monthly limit", "monthly quota")
_DAILY_HINTS = ("per day", "per-day", "daily free allocation", "daily limit", "daily quota")
DAILY_QUOTA_RETRY = 3600.0
# Google AI Studio refuses whole regions with a 400.
_LOCATION_HINTS = ("location is not supported", "unsupported_country", "not available in your country")
# Google puts the server-suggested wait in the JSON body: "retryDelay": "36s"
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


def classify(status: int, headers: Optional[Dict[str, str]], body: str, provider: str) -> ProviderError:
    """Map an HTTP error response to the right exception.

    Nothing here aborts a call on its own: even an unrecognised 4xx becomes a
    ``BadRequest`` that fails over, and the engine raises only once several
    providers reject the same request (see ``_engine.should_raise``)."""
    retry_after = parse_retry_after((headers or {}).get("retry-after"))
    if retry_after is None and body:
        m = _RETRY_DELAY_RE.search(body)
        if m:
            retry_after = float(m.group(1))
    msg = (body or "")[:300]
    low = (body or "").lower()
    if status in (400, 401, 403) and any(h in low for h in _AUTH_HINTS):
        return AuthError(provider, status, msg)
    if status == 403 and any(h in low for h in _MODERATION_HINTS):
        return BadRequest(provider, status, msg)
    if status == 403 and any(h in low for h in _PLAN_HINTS):
        return ModelNotFound(provider, status, msg, gone=True)  # not on this account's plan
    if status in (401, 403):
        return AuthError(provider, status, msg)
    if status == 402:
        return QuotaExhausted(provider, status, msg)
    if status == 429:
        if any(h in low for h in _MONTHLY_HINTS):
            return QuotaExhausted(provider, status, msg)
        if retry_after is None and any(h in low for h in _DAILY_HINTS):
            retry_after = DAILY_QUOTA_RETRY
        return RateLimited(provider, status, msg, retry_after=retry_after)
    if status in _TRANSIENT_STATUS or 500 <= status <= 599:
        return Transient(provider, status, msg, retry_after=retry_after)
    if status in _GONE_STATUS:
        # OpenRouter 404s "No endpoints found that support tool use": the model
        # is fine, it just can't do what this request asks — don't bench it
        gone = not any(h in low for h in _CAPABILITY_HINTS)
        retired = gone and (status == 410 or any(h in low for h in _RETIRED_HINTS))
        return ModelNotFound(provider, status, msg, gone=gone, retired=retired)
    if status in (400, 413, 422):
        if "model" in low and any(h in low for h in _GONE_HINTS):
            return ModelNotFound(provider, status, msg, gone=True, retired=any(h in low for h in _RETIRED_HINTS))
        if status == 413 or any(h in low for h in _TOO_BIG_HINTS) or "model" in low:
            # this model can't take *this* request (too big, unsupported
            # parameter, ...) — another model or provider may
            return ModelNotFound(provider, status, msg)
    if any(h in low for h in _LOCATION_HINTS):
        return AuthError(provider, status, msg)  # the key is unusable from here
    return BadRequest(provider, status, msg)
