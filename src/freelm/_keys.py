"""Per-key runtime state: breaker + rpm bucket + daily quota + cooldowns."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ._breaker import CircuitBreaker
from ._ratelimit import TokenBucket

DAY = 86400.0


@dataclass
class KeyState:
    key: str
    tier: str = "free"
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    bucket: Optional[TokenBucket] = None
    rpd: Optional[int] = None           # requests-per-day cap (None = unknown/unlimited)
    rpd_used: int = 0
    rpd_reset: float = 0.0              # monotonic ts at which the daily counter rolls
    cooldown_until: float = 0.0
    disabled: bool = False              # hard-off after auth failure
    ewma_latency: float = 0.0
    last_error: Optional[str] = None

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
        """A rough 'how much headroom' score for quota-aware routing."""
        self._roll_daily(now)
        daily = float("inf") if self.rpd is None else float(max(0, self.rpd - self.rpd_used))
        burst = self.bucket.peek(now) if self.bucket is not None else float("inf")
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
        k = self.key
        return (k[:6] + "..." + k[-4:]) if len(k) > 12 else "***"


def new_key_state(key: str, *, tier: str, rpm: Optional[float], rpd: Optional[int]) -> KeyState:
    return KeyState(
        key=key,
        tier=tier,
        breaker=CircuitBreaker(),
        bucket=TokenBucket(rate_per_min=rpm) if rpm else None,
        rpd=rpd,
    )
