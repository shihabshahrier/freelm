from freelm._breaker import CircuitBreaker
from freelm._keys import new_key_state
from freelm._ratelimit import TokenBucket
from freelm.registry import ModelSpec, resolve_models


def test_token_bucket_consume_and_refill():
    b = TokenBucket(rate_per_min=60, capacity=2)  # 1 token/sec
    now = 100.0
    assert b.consume(1, now) is True
    assert b.consume(1, now) is True
    assert b.consume(1, now) is False        # empty
    assert round(b.time_until(1, now), 2) == 1.0
    assert b.consume(1, now + 1.0) is True   # refilled one


def test_circuit_breaker_opens_and_half_opens():
    cb = CircuitBreaker(fail_threshold=2, cooldown=10)
    now = 0.0
    assert cb.allow(now)
    cb.on_failure(now)
    assert cb.state == "closed"
    cb.on_failure(now)
    assert cb.state == "open"
    assert cb.allow(now) is False            # still cooling
    assert cb.allow(now + 11) is True        # half-open after cooldown
    cb.on_success()
    assert cb.state == "closed"


def test_keystate_daily_quota_and_reset():
    k = new_key_state("k", tier="free", rpm=None, rpd=2)
    now = 1000.0
    assert k.ready(now)
    assert k.reserve(now)
    assert k.reserve(now)
    assert k.ready(now) is False             # daily exhausted
    assert k.ready(now + 86400 + 1) is True  # rolled over


def test_keystate_cooldown_and_disable():
    k = new_key_state("k", tier="free", rpm=None, rpd=None)
    now = 0.0
    k.cooldown_until = now + 30
    assert k.ready(now) is False
    assert k.ready(now + 31) is True
    k.disabled = True
    assert k.ready(now + 31) is False
    assert k.wait_time(now + 31) is None


def test_registry_resolution():
    models = [
        ModelSpec("big/model", ("chat", "large")),
        ModelSpec("small/model", ("chat", "small", "fast")),
    ]
    assert resolve_models(models, "auto") == ["big/model", "small/model"]
    assert resolve_models(models, "chat:large") == ["big/model"]
    assert resolve_models(models, "fast") == ["small/model"]
    assert resolve_models(models, "small/model") == ["small/model"]   # exact id
    assert resolve_models(models, "vendor/unknown-xyz") == ["vendor/unknown-xyz"]  # passthrough


def test_registry_passthrough_with_colon_suffix():
    # ids like "vendor/model:free" that are NOT in the list must pass through
    # verbatim — never silently fan out to the whole chat list.
    models = [ModelSpec("big/model:free", ("chat", "large"))]
    assert resolve_models(models, "moonshotai/kimi-k2:free") == ["moonshotai/kimi-k2:free"]
    assert resolve_models(models, "big/model:free") == ["big/model:free"]  # exact still wins


def test_apply_success_zero_latency_keeps_ewma():
    from freelm._engine import apply_success
    from freelm.strategy import Candidate

    k = new_key_state("k", tier="free", rpm=None, rpd=None)
    k.ewma_latency = 100.0
    cand = Candidate(provider=None, key=k, model="m")
    apply_success(cand, 0.0)            # "no sample" (e.g. empty stream)
    assert k.ewma_latency == 100.0      # must not decay toward 0
    apply_success(cand, 200.0)
    assert k.ewma_latency == 100.0 * 0.7 + 200.0 * 0.3
