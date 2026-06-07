"""Exponential backoff with full jitter."""
from __future__ import annotations

import random


def compute_delay(attempt: int, base: float = 0.5, factor: float = 2.0, cap: float = 30.0, jitter: bool = True) -> float:
    """Delay (seconds) for a given retry attempt (1-based)."""
    attempt = max(1, attempt)
    raw = min(cap, base * (factor ** (attempt - 1)))
    if jitter:
        return random.uniform(0.0, raw)
    return raw
