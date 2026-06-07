"""Token bucket for requests-per-minute pacing. Time injected for testability."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenBucket:
    rate_per_min: float
    capacity: Optional[float] = None
    tokens: Optional[float] = None
    updated: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.capacity is None:
            self.capacity = max(1.0, float(self.rate_per_min))
        if self.tokens is None:
            self.tokens = self.capacity

    def _refill(self, now: float) -> None:
        if self.updated == 0.0:
            self.updated = now
            return
        dt = now - self.updated
        if dt <= 0:
            return
        self.tokens = min(self.capacity, self.tokens + dt * (self.rate_per_min / 60.0))  # type: ignore[operator]
        self.updated = now

    def peek(self, now: float) -> float:
        with self._lock:
            self._refill(now)
            return self.tokens  # type: ignore[return-value]

    def consume(self, n: float, now: float) -> bool:
        with self._lock:
            self._refill(now)
            if self.tokens >= n:  # type: ignore[operator]
                self.tokens -= n  # type: ignore[operator]
                return True
            return False

    def time_until(self, n: float, now: float) -> float:
        with self._lock:
            self._refill(now)
            if self.tokens >= n:  # type: ignore[operator]
                return 0.0
            deficit = n - self.tokens  # type: ignore[operator]
            return deficit / (self.rate_per_min / 60.0)
