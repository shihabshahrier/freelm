"""Per-key circuit breaker. Time is injected (monotonic seconds) for testability."""
from __future__ import annotations

from dataclasses import dataclass

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    fail_threshold: int = 4
    cooldown: float = 30.0
    state: str = CLOSED
    failures: int = 0
    opened_at: float = 0.0

    def allow(self, now: float) -> bool:
        """May a request go through right now?"""
        if self.state == OPEN:
            if now - self.opened_at >= self.cooldown:
                self.state = HALF_OPEN
                return True
            return False
        return True

    def on_success(self) -> None:
        self.failures = 0
        self.state = CLOSED

    def on_failure(self, now: float) -> None:
        self.failures += 1
        if self.state == HALF_OPEN or self.failures >= self.fail_threshold:
            self.state = OPEN
            self.opened_at = now

    def time_until_half_open(self, now: float) -> float:
        if self.state != OPEN:
            return 0.0
        return max(0.0, self.cooldown - (now - self.opened_at))
