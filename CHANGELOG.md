# Changelog

All notable changes to `freelm` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [0.2.3] - 2026-06-10

Applies to Python 0.2.3 and the JS/TS package 0.1.1 (the two track each other).

### Fixed
- **Model passthrough**: a concrete model id containing `:` that wasn't in the
  provider's current list (e.g. a new OpenRouter `:free` id) silently resolved
  to *all* chat models instead of the requested one. Unknown ids now pass
  through verbatim, as documented.
- **402 handling**: out-of-credit responses (e.g. OpenRouter below the free
  threshold) aborted the whole call. Now classified as `QuotaExhausted`: the
  key is disabled and the call fails over, like an auth error.
- **OpenAI compat shim** is actually drop-in now: OpenAI-SDK constructor
  arguments (`api_key`, `base_url`, ... / `{ apiKey, baseURL, ... }` in JS) are
  accepted and ignored, and `stream=True` yields `chat.completion.chunk`-shaped
  objects instead of being silently swallowed.
- **Latency stats**: streaming successes recorded 0 ms and decayed the latency
  EWMA toward zero, skewing the `latency` strategy. Streams now record
  time-to-first-token, and zero-latency samples are ignored.
- **OpenRouter 429 scope**: a bare "temporarily" in the body no longer marks a
  429 as model-scoped — account-wide limits cool the key again.
- **Discovery cache**: a cached `/models` list that filters down to zero usable
  chat models no longer blocks the live refetch until TTL expiry.
- **JS timeouts**: response-body reads and SSE streams are now covered by the
  timeout (per-chunk inactivity, mirroring httpx); discovery fetches time out
  after 15 s; an abandoned stream cancels the underlying connection.
- Google fallback models refreshed (`gemini-2.5-flash` family; retired
  `gemini-1.5-flash` dropped).

### Changed
- Version is single-sourced (`freelm._version` / `js/src/version.ts`, enforced
  by a test in JS); the User-Agent strings derive from it.
- CI lints with ruff; the PyPI release workflow runs tests before publishing.

[0.2.3]: https://github.com/shihabshahrier/freelm/releases/tag/v0.2.3

## [0.2.2] - 2026-06-07

### Fixed
- Discovery filters more non-chat models (image-gen + audio/TTS) that some
  providers list without modality metadata (imagen, veo, dall-e, orpheus, ...).
- `auto` deprioritizes reasoning models even when `/models` has no metadata, by
  detecting them from the model id (gpt-oss, deepseek-r1, magistral, ...), so a
  default call leads with a plain instruct model.

### Notes
- **Free-only:** this library covers free-tier providers only. **Groq** (`gsk_…`)
  is supported; **xAI Grok** (`xai-…`) is a different, *paid* service and is
  intentionally not included.
- Live end-to-end tested all **six** free providers (OpenRouter, Google AI Studio,
  NVIDIA NIM, Groq, Cerebras, Mistral): chat + streaming + live model discovery.

[0.2.2]: https://github.com/shihabshahrier/freelm/releases/tag/v0.2.2

## [0.2.1] - 2026-06-07

### Fixed
- Verified provider model IDs against official docs; corrected stale Mistral
  fallback IDs (`open-mistral-nemo` → `-latest` aliases). Groq, Cerebras, and
  Mistral now run live `/models` discovery so their model lists self-correct at
  runtime — the hardcoded lists are offline fallbacks only.
- Discovery filters out non-chat models (whisper / TTS / embedding / rerank /
  guard / OCR) that some providers list without modality metadata.

[0.2.1]: https://github.com/shihabshahrier/freelm/releases/tag/v0.2.1

## [0.2.0] - 2026-06-07

### Added
- **Streaming**: `FreeLLM.stream()` and `AsyncFreeLLM.astream()` yield content
  deltas (SSE), normalized across providers, with failover *before* the first
  token (no mid-stream switching once tokens flow).
- **Three more free providers** (all OpenAI-compatible), with free-tier limits
  verified 2026-06: **Groq** (30 RPM / 14.4K req-day), **Cerebras** (~30 RPM /
  1M tokens-day, 8K ctx), **Mistral** (2 RPM / 500K TPM / 1B-month). Env config:
  `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `MISTRAL_API_KEY`.
- `.env.example` and an env-only `examples/e2e_smoke.py` (keys are never inlined).

### Changed
- Default `auto` model order leads with fast plain instruct models; giant
  (>150B) and reasoning models are deprioritized (they were slow and verbose
  for a default — surfaced by live E2E testing).

[0.2.0]: https://github.com/shihabshahrier/freelm/releases/tag/v0.2.0

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
