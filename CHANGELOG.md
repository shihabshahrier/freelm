# Changelog

All notable changes to `freelm` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [0.5.2] - 2026-10-10

### Fixed
- **Gemini for new accounts.** Gemini 2.0 is shut down and 2.5 answers new
  accounts with 404 "no longer available to new users". The built-in Gemini
  list is now gemini-3.5-flash-lite, 3.1-flash-lite, flash-lite-latest,
  3.5-flash, 3.7-flash and flash-latest, each verified live on the free tier.
  Free Gemini quotas are per model, so the key-wide daily heuristic is gone.
- **Gemini daily-quota 429s** name a `PerDay` quota id but suggest a ~30 s
  `retryDelay`; freelm now rests that model for an hour instead of retrying
  after 30 s into a spent daily cap.
- "Invalid Auth key." and `ACCESS_TOKEN_TYPE_UNSUPPORTED` (Gemini's newer
  `AQ.` keys) are treated as a dead key.
- Groq's built-in fallback model is qwen/qwen3.8-27b.

### Docs
- README links an error reference: one guide per common free-tier error
  (https://shahriarlabs.com/products/freelm/errors/).

## [0.5.1] - 2026-10-10

Packaging, docs and one header change.

### Added
- `benchmarks/failover.py` and `benchmarks/failover.mjs`: a reproducible
  provider-failure simulation (fake providers on localhost, real HTTP, default
  settings). `--installed` runs it against any installed release, so the
  0.4.0 vs 0.5.0 numbers in `benchmarks/README.md` can be checked by anyone.
- README: setup for OpenCode, OpenClaw and Hermes Agent with `freelm serve`.

### Changed
- OpenRouter requests also send `X-OpenRouter-Title` (the current name of
  `X-Title`, which is kept) and `X-OpenRouter-Categories:
  programming-app,general-chat`, so freelm's page on openrouter.ai/apps is
  categorised. Override any of them with `extra_headers` / `extraHeaders`.
- PyPI and npm listings: shorter keyword-first summaries, the PyPI homepage
  points at the canonical product page, SPDX license expression (PEP 639),
  extra classifiers, and an FAQ in the npm README.

## [0.5.0] - 2026-10-10

### Added — failover that doesn't wait
- **Hedged attempts** (`hedge=True`, the default): when an attempt is still
  running after a few seconds — 3x that key's usual latency, clamped (streams:
  1.5–6 s, 3 s unmeasured; whole answers: 4–12 s, 6 s unmeasured) — the next
  candidate starts in parallel and the first answer wins. The loser is
  cancelled and remembered as slow. Before, a provider that hung (or a host that
  never accepted the connection, or a stream that stalled before its first
  token) held the call for the whole 60 s deadline and the call **failed**
  without trying the healthy provider behind it. Measured on real HTTP: hang
  → 6.3 s (was a 60 s failure), stalled stream → 3.3 s. `hedge=False` restores
  one attempt at a time; a number sets a fixed delay. New event kind: `hedge`.
- **`smart` routing, the new default strategy**: priority tiers first, then the
  fastest measured provider. A provider without a fresh sample counts as
  typical (2 s), so unknown providers get tried and measured; samples expire
  after 10 minutes, so a provider that was slow once is reconsidered. A slow
  provider (8 s) used to cost 8 s on every call; now only the first.
  `strategy="priority"` keeps the old fixed order.

### Changed
- The sync client discovers every provider's models in parallel (the async
  clients already did): first call with 6 catalogs to fetch 4.6 s → 0.3 s.
- Python: a host that won't accept the connection fails within 10 s (connect
  timeout), like Node's fetch, instead of using the whole call deadline.

### Added — three more free providers (beta: not yet verified with live keys)
- **Cloudflare Workers AI** (`CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`):
  10,000 Neurons/day free on every account (a few hundred chats), with curated
  Free-plan `@cf/...` models — Llama 4 Scout, Mistral Small 3.1, gpt-oss,
  Qwen 3.8, Gemma 4, GLM-4.7-Flash, Llama 3.3 70B. The account id is part of
  the endpoint, so `CloudflareWorkersAI(token, account_id=...)` /
  `{ accountId }` requires it; from the environment, a token without one is
  skipped with a notice (and `freelm doctor` says what's missing) instead of
  failing the whole setup. Requests carry `max_tokens: 4096` unless you set one
  (many Workers AI models otherwise stop at 256 tokens), and a spent daily
  allowance rests the key for an hour instead of being retried every minute.
- **Z.ai** (`ZAI_API_KEY`): GLM-4.7-Flash, GLM-4.5-Flash and GLM-4.6V-Flash
  (vision) — Z.ai's free models. Every other GLM model is billed against the
  account balance, so the provider is free-only with an explicit list: a paid id
  is refused (`ConfigError`) instead of spending credit.
- **Cohere** (`COHERE_API_KEY` or `CO_API_KEY`) through its OpenAI compatibility
  API: Command A+, Command A (incl. Reasoning and Vision) and Command R7B on a
  free trial key (20 requests/minute per model, 1,000 calls/month,
  non-commercial use). A per-minute 429 benches just that model; the monthly
  cap disables the key.
- `freelm.config.build_provider()` / `buildProvider()` builds one provider from
  the environment and `env_vars()` / `envVars()` lists the variables it needs;
  `ProviderEnv.option_vars` / `optionVars` map extra constructor options to env
  vars. `freelm doctor` now prints every variable a provider needs
  (`export CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=...`) and shows
  Cloudflare's error messages.
- `Provider.adapt_payload()` / `adaptPayload()`: a hook to adjust the request
  body for a provider's quirks (used by Cloudflare for `max_tokens`).

### Fixed
- **Quota 429s no longer hammer a spent key every minute.** A 429 that says the
  quota is gone for the day (Cloudflare's daily Neurons, OpenRouter's
  `free-models-per-day`) and gives no retry time now rests the key for an hour;
  one that says it's gone for the month (Cohere trial keys) disables the key,
  like a 402.
- A 403 saying a model needs a paid plan (Cloudflare's Workers Paid-only
  models) benches that model for the key instead of disabling the key, and a
  400 "no such model" benches the model instead of retrying it on every call.
- A numeric token sent as a JSON number (`"content": 6`, seen from Workers AI)
  is kept as text — `stream()` used to drop it silently.

## [0.4.0] - 2026-10-09

Python and the JS/TS package both move to 0.4.0 (versions are aligned from now on).

### Fixed — the free tiers changed under us
- **A retired model no longer kills the call.** NVIDIA NIM answers `410 Gone`
  for end-of-life models (all of NIM's previous defaults since 2026-08-26), and
  any status freelm didn't recognise was raised as a caller bug — so with a
  NIM key configured, `chat()`/`stream()` failed even when five other providers
  were healthy. 404/410 (and 400s saying "decommissioned", "no longer
  available", ...) now mean "model gone": the provider *benches* that model for
  an hour and the call fails over. Benched models keep their place in the
  breadth-first interleave, so a provider full of dead models can't starve the
  others.
- **No status aborts a call by itself any more.** All 5xx (incl. Cloudflare
  52x and 501) are transient; 413 / "context too long" try a model with room;
  Google's 400 `API_KEY_INVALID` / location errors disable the key; OpenRouter's
  moderation 403 no longer disables the key. Any other 4xx is a `BadRequest`
  (new) that fails over to the next *provider* and is raised only once two
  providers reject the same request — free tiers disagree about which
  parameters they accept. A rejected request no longer trips the breaker.
- **Concrete model ids work next to OpenRouter.** `model="gemini-2.5-flash"`
  (or any Groq/NIM/Mistral id) raised OpenRouter's free-guard `ConfigError`
  whenever an OpenRouter key was configured. Concrete ids now go only to the
  providers that list them; the guard skips its provider instead of failing
  the call (it still raises when nobody else can serve).
- **Errors delivered with HTTP 200** — `{"error": ...}` bodies, OpenRouter's
  mid-stream error frames, non-JSON 200s — fail over instead of returning
  empty text or crashing with a JSON error.
- **Per-model limits are per-model**: Gemini and Groq 429s bench just that
  model *for that key* (other models, and other keys, keep serving); Gemini's
  "high demand" 503s and OpenRouter's upstream throttles bench the model on
  every key; a model retired for everyone (410 / "end of life") is benched
  provider-wide while "no access" 404s only affect the key that got them;
  Google's `retryDelay` is honoured; `Retry-After: 0` no longer means 60 s.
  A 402 naming a model benches the model, not the key.
- **Fresh default models for every provider** (all of June's fallbacks were
  dead): Gemini verified live on the free tier, Groq's named successors after
  the 2026-08-16 shutdown, NIM/Cerebras/OpenRouter from their live catalogs.
- **`wait=True` no longer spins** until the deadline when every candidate was
  tried or benched; each attempt is bounded by the call's remaining time;
  discovery uses a 10 s timeout, runs once for concurrent first calls (async),
  tries the next key when the first is dead, and `refresh_models()` really
  re-fetches.
- **Discovery tagging**: size words match whole tokens ("gemini" is not
  "mini"), Groq/Mistral metadata (context window, tool/vision capability) is
  read, Google's `models/` prefix is stripped, safety/reward/parse models are
  filtered, and paid OpenRouter entries are never picked for an alias on a
  free-only provider. An explicit `models=[...]` list is no longer overwritten
  by discovery. `chat:tools` / `vision` never fall back to incapable models.
- **Secrets**: `repr()` / `util.inspect` / `JSON.stringify` of keys, candidates
  and errors show masked keys only.
- **Persistence** (`persist=True`): a daily counter from an expired window is
  reset on load, concurrent writers use unique temp files, bad field types are
  tolerated, and a persisted `disabled` flag expires after 24 h.
- **Messages**: OpenAI-SDK message objects (`model_dump()`) are accepted and
  null fields (`"tool_calls": null`) are dropped before sending; content-part
  arrays flatten into `.text`.
- **JS**: a stream consumer slower than `timeout` could hang forever (the
  inactivity timer ran while the generator was paused); timers now only run
  around network reads and never keep the process alive. `[DONE]` closes the
  connection. Caller `signal` (AbortSignal) is supported and no longer leaks
  into the request body. `freelm/compat` in CommonJS no longer bundles a second
  copy of the library (broken `instanceof`). The package imports no `node:`
  built-ins statically, so it bundles for Workers/edge/browsers; disk cache and
  persistence switch on where Node built-ins exist.

### Added
- **Works with zero keys (CLI) — keyless providers.** `Kilo` (Kilo Gateway: free
  routes work without a key, ~200 req/hour per IP; `KILO_API_KEY` optional,
  free-only guard on) and `OVHcloud` (AI Endpoints anonymous tier, 2 req/min per
  IP per model; keys are pay-as-you-go so they are never used). The CLI falls
  back to them when no keys are configured, with a notice; the library only on
  request (`from_env(keyless=True | "auto")`, `FREELM_KEYLESS=1|auto|0`) so
  prompts are never sent to them silently.
- **OpenRouter `openrouter/free`** router as the last fallback model; the free
  guard now accepts any catalog entry priced at zero (and Kilo's `isFree`).
- **`freelm serve`** — the router as a local OpenAI-compatible endpoint
  (`/v1/chat/completions` with SSE streaming, `/v1/models`, `/health`), zero
  dependencies in both languages. Point Cursor, Cline, Continue, Open WebUI,
  n8n, LangChain or the Vercel AI SDK at `http://127.0.0.1:4000/v1`. Optional
  `--api-key`, `--cors`; unknown model ids (a tool's default `gpt-4o`) fall back
  to `auto` unless `--no-fallback`. Also `freelm.serve()` / `serve()` in code.
- **Docker image** for `freelm serve` (`ghcr.io/shihabshahrier/freelm`, amd64 + arm64), published on release tags.
- **`freelm doctor`** — live-checks every configured key with one tiny request
  and says exactly what's wrong and where to get a new free key (`--json` too).
- **`stream_chunks()` / `astream_chunks()` / `streamChunks()`** — raw
  `chat.completion.chunk` streaming with tool-call deltas, finish reasons and
  usage, failover still invisible before the first output.
- `reset_keys()` / `resetKeys()`; `NoProvidersAvailable.status` (one line per
  provider explaining why it's unusable, also in the message).
- **OpenAI compat shim fidelity**: attribute access all the way down
  (`message.tool_calls[0].function.arguments`), `created`/`id`, `model_dump()` /
  `to_dict()` / `model_dump_json()`, streams as context managers with tool-call
  deltas and finish reasons, `models.list()`, `extra_body`, `with` blocks;
  JS: millisecond `timeout`s from openai-node, `create(body, { signal })`,
  `stream.controller`, `models.list()` (awaitable and async-iterable).
- OpenRouter app attribution headers (`HTTP-Referer`, `X-Title`).

### Changed
- **Cerebras is no longer permanently free** (trial credits, card required as of
  2026-10): still supported for existing keys, documented as such.
- Groq free-tier defaults follow its current per-model limits (30 RPM, 1K RPD).
- JS requires Node >= 20 (Node 18 and 20 are end-of-life upstream; CI covers
  20/22/24). Dev tooling: vitest 4 (clears the dev-only audit advisories).
- CI: ruff's rule selection is pinned in `pyproject.toml` (ruff 0.16 widened
  its defaults and turned CI red without a code change); GitHub Actions bumped
  to current majors; release workflows support PyPI/npm Trusted Publishing.

[0.4.0]: https://github.com/shihabshahrier/freelm/releases/tag/v0.4.0

## [0.3.0] - 2026-06-10

Applies to Python 0.3.0 and the JS/TS package 0.2.0 (the two track each other).

### Added
- **Model priority, three ways**: `ModelSpec(priority=)` for static lists;
  provider `prefer=[...]` (exact id or substring) to bias *discovered* lists
  without replacing them; per-call ordered fallback chains
  (`chat(msgs, model=["vendor/id", "chat:fast"])`).
- **Provider priority everywhere**: `priority=` is now the tiebreak in
  `quota_aware` and `latency` and the baseline order for `round_robin`.
- **Free-only guard**: OpenRouter ships `free_only=True` — passing a paid
  (non-`:free`) model id raises `ConfigError` with the opt-out
  (`free_only=False`) instead of silently billing. New README section
  "When can freelm cost money?".
- **Capability aliases**: `chat:tools`, `vision`, `reasoning` route to models
  tagged by discovery; `tools`/`tool_choice`/`response_format` pass through and
  `ChatResponse.tool_calls` (JS: `toolCalls`) surfaces function calls.
- **Observability**: `FreeLLM(on_event=...)` / `{ onEvent }` emits
  `attempt | success | error | wait | discovery` events with masked keys;
  a raising callback never breaks the call.
- **Persistent quota state** (opt-in `persist=True` / `FREELM_PERSIST=1`):
  rpd counters, cooldowns, and disabled keys survive restarts via
  `~/.cache/freelm/state.json` (0600, atomic, key *hashes* only — schema shared
  between the Python and JS packages).
- **CLI**: `freelm chat|models|health` (Python entry point; `npx freelm` in JS).
  Zero new dependencies in both languages.

[0.3.0]: https://github.com/shihabshahrier/freelm/releases/tag/v0.3.0

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
