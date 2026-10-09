"""Per-key runtime state: breaker + rpm bucket + daily quota + cooldowns."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from ._breaker import CircuitBreaker
from ._ratelimit import TokenBucket

DAY = 86400.0
# Placeholder "key" for keyless providers (no Authorization header is sent).
ANONYMOUS = "anonymous"
# stand-in for "unlimited" daily quota so it ranks high but stays finite/comparable
UNLIMITED = 100_000.0


@dataclass
class KeyState:
    key: str = field(repr=False)  # never in reprs/tracebacks — see __repr__
    tier: str = "free"
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    bucket: Optional[TokenBucket] = None
    rpd: Optional[int] = None           # requests-per-day cap (None = unknown/unlimited)
    rpd_used: int = 0
    rpd_reset: float = 0.0              # monotonic ts at which the daily counter rolls
    cooldown_until: float = 0.0
    disabled: bool = False              # hard-off after auth failure
    disabled_since_wall: float = 0.0    # wall-clock ts it was disabled (persistence TTL)
    ewma_latency: float = 0.0
    last_error: Optional[str] = None
    # model id -> monotonic ts until which this key must not use that model
    # (its own per-model quota hit, no access to the model, ...)
    model_until: Dict[str, float] = field(default_factory=dict, repr=False)

    # -- per-model benches (this key only) ------------------------------
    def bench_model(self, model: str, until: float) -> None:
        self.model_until[model] = max(until, self.model_until.get(model, 0.0))

    def model_wait(self, model: str, now: float) -> float:
        until = self.model_until.get(model)
        if until is None:
            return 0.0
        if now >= until:
            del self.model_until[model]
            return 0.0
        return until - now

    def model_ready(self, model: str, now: float) -> bool:
        return self.model_wait(model, now) == 0.0

    # -- daily window ----------------------------------------------------
    def _roll_daily(self, now: float) -> None:
        if self.rpd is None:
            return
        if self.rpd_reset == 0.0:
            self.rpd_reset = now + DAY
        elif now >= self.rpd_reset:
            self.rpd_used = 0
            self.rpd_reset = now + DAY

    # -- gating ----------------------------------------------------------
    def ready(self, now: float) -> bool:
        if self.disabled:
            return False
        if now < self.cooldown_until:
            return False
        if not self.breaker.allow(now):
            return False
        self._roll_daily(now)
        if self.rpd is not None and self.rpd_used >= self.rpd:
            return False
        if self.bucket is not None and self.bucket.peek(now) < 1:
            return False
        return True

    def reserve(self, now: float) -> bool:
        """Consume one rpm token + one daily slot just before firing a request."""
        self._roll_daily(now)
        if self.bucket is not None and not self.bucket.consume(1, now):
            return False
        self.rpd_used += 1
        return True

    def remaining(self, now: float) -> float:
        """Headroom score for quota-aware routing: current rpm tokens bounded by
        daily quota left. Not-ready keys (cooling/disabled/exhausted) score 0 so
        they don't attract traffic. 'Unlimited' daily is capped, not infinite."""
        if not self.ready(now):
            return 0.0
        daily = UNLIMITED if self.rpd is None else float(max(0, self.rpd - self.rpd_used))
        burst = self.bucket.peek(now) if self.bucket is not None else UNLIMITED
        return min(daily, burst)

    def wait_time(self, now: float) -> Optional[float]:
        """Seconds until this key could be ready again, or None if permanently off."""
        if self.disabled:
            return None
        waits = []
        if now < self.cooldown_until:
            waits.append(self.cooldown_until - now)
        waits.append(self.breaker.time_until_half_open(now))
        self._roll_daily(now)
        if self.rpd is not None and self.rpd_used >= self.rpd:
            waits.append(max(0.0, self.rpd_reset - now))
        if self.bucket is not None and self.bucket.peek(now) < 1:
            waits.append(self.bucket.time_until(1, now))
        return max(waits) if waits else 0.0

    def masked(self) -> str:
        return mask_key(self.key)

    def __repr__(self) -> str:
        return (
            f"KeyState(key={self.masked()!r}, tier={self.tier!r}, disabled={self.disabled}, "
            f"rpd_used={self.rpd_used}, last_error={self.last_error!r})"
        )


def mask_key(key: str) -> str:
    """How a key is shown anywhere (events, health, errors, doctor) — never raw."""
    if key == ANONYMOUS:
        return "(keyless)"
    return (key[:6] + "..." + key[-4:]) if len(key) > 12 else "***"


def new_key_state(key: str, *, tier: str, rpm: Optional[float], rpd: Optional[int]) -> KeyState:
    return KeyState(
        key=key,
        tier=tier,
        breaker=CircuitBreaker(),
        bucket=TokenBucket(rate_per_min=rpm) if rpm else None,
        rpd=rpd,
    )
