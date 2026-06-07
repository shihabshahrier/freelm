# Changelog

All notable changes to `freelm` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [0.1.1] - 2026-06-07

### Fixed
- **Failover starvation**: candidates are now interleaved breadth-first across
  providers, so a provider with many throttled free models can no longer starve
  the others. Default `max_attempts` raised 6 → 12; added an overall per-call
  deadline derived from `timeout`.
- `wait=True` now retries keys that recover during the sleep (previously it
  skipped any already-tried key).
- `quota_aware` no longer treats unlimited daily quota as infinite, and scores
  cooling/disabled keys as 0 headroom.
- Refund the daily-quota slot when a request fails with 404 `ModelNotFound`.

### Docs
- Documented tuning knobs (`max_attempts` / `timeout` / `wait` / `priority` / ...),
  strategy semantics, error hierarchy, response + `health()` reference, and a
  concurrency note.

## [0.1.0] - 2026-06-07

Initial release.

### Added
- `FreeLLM` (sync) and `AsyncFreeLLM` (async) always-up chat clients.
- Providers (OpenAI-compatible HTTP): OpenRouter, Google AI Studio (Gemini), NVIDIA NIM.
- Fault tolerance: per-key circuit breaker, cross-provider failover, retry classification
  (429 cooldown/rotate, 5xx/timeout backoff, 401 key-disable, model errors → next model).
- Model-scoped vs key-scoped 429 handling (OpenRouter free models throttle per-model upstream).
- Quota guard: per-key requests/minute token bucket + requests/day counter.
- Routing strategies: `priority`, `round_robin`, `quota_aware`, `latency`.
- Virtual models (`auto`, `chat:fast`, `chat:large`, ...) resolved per provider.
- Dynamic model discovery via `GET /models` with disk cache (TTL, `0600`) and
  live → cache → hardcoded fallback. `list_free_models()` helper.
- OpenAI drop-in shim: `freelm.compat.OpenAI` / `AsyncOpenAI`.
- `FreeLLM.from_env()` config from environment; `llm.health()` introspection.

[0.1.1]: https://github.com/shihabshahrier/freelm/releases/tag/v0.1.1
[0.1.0]: https://github.com/shihabshahrier/freelm/releases/tag/v0.1.0
