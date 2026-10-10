# freelm — agent guide

Free, always-up LLM client + gateway pooling free-tier providers (OpenRouter,
Google AI Studio, NVIDIA NIM, Groq, Cerebras, Mistral, Z.ai, Cohere, Cloudflare
Workers AI; keyless Kilo/OVHcloud on request) behind one
OpenAI-compatible API with key rotation, cross-provider failover, circuit
breaking, quota-aware routing, and live model discovery.

**Dual implementation, one repo:**
- Python package `freelm` → `src/freelm/` (PyPI), tests in `tests/`
- TypeScript package `freelm` → `js/src/` (npm), tests in `js/test/`

## The parity rule (most important)

The TS port mirrors the Python implementation file-for-file and
behavior-for-behavior. **Any behavior change must land in BOTH languages in the
same commit, with tests in both.** File mapping is 1:1:

| Python (`src/freelm/`)  | TypeScript (`js/src/`) |
|-------------------------|------------------------|
| `client.py` (sync+async)| `client.ts` (async only) |
| `_engine.py`            | `engine.ts`            |
| `strategy.py`           | `strategy.ts`          |
| `_keys.py`              | `keys.ts`              |
| `_breaker.py` / `_ratelimit.py` / `_backoff.py` | `breaker.ts` / `ratelimit.ts` / `backoff.ts` |
| `errors.py`             | `errors.ts`            |
| `registry.py`           | `registry.ts`          |
| `discovery.py` / `_cache.py` | `discovery.ts` / `cache.ts` |
| `config.py`             | `config.ts`            |
| `providers/*.py`        | `providers/*.ts`       |
| `compat/openai.py` + `types_compat.py` | `compat/openai.ts` |
| `_state.py`             | `state.ts`             |
| `_cli.py` + `__main__.py` | `cli.ts` (+ `bin/freelm.mjs`) |
| `server.py`             | `server.ts`            |
| `_version.py`           | `version.ts`           |
| —                       | `runtime.ts` (TS-only: env / Node built-ins without static `node:` imports) |

## Commands

```bash
# Python (venv at .venv, Python >= 3.9)
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q                      # tests
.venv/bin/ruff check src tests examples  # lint (CI-enforced)

# TypeScript (Node >= 20)
cd js && npm ci
npm test            # vitest
npm run typecheck   # tsc --noEmit
npm run build       # tsup -> dist/ (esm+cjs+dts)
```

Live smoke test (needs real keys in env, never hardcoded): run `freelm doctor`
first (which keys work today), then `examples/e2e_smoke.py` /
`js/examples/e2e.mjs`.

## Architecture (both languages)

- **Layering:** `engine` + `strategy` are pure (no I/O, time injected as
  monotonic seconds) — all orchestration decisions live there so the sync and
  async Python clients and the TS client share identical logic. Only
  `client.*` performs HTTP (httpx in Python, built-in `fetch` in TS).
- **Candidate loop:** a candidate is one (provider, key, model). Per call:
  order candidates by strategy, interleave breadth-first across providers
  (rank 0 of every provider before any rank 1), skip `tried`, `reserve()` a
  token, fire, classify the outcome, repeat up to `max_attempts` within a
  `timeout` deadline. The loop is a **race** (`_race`/`race`): one attempt at a
  time, plus at most one parallel *hedge* when the running attempt outlives
  `engine.hedge_delay` (3x the key's latency EWMA, clamped; `HEDGE_STREAM` /
  `HEDGE_CHAT`); first success wins, losers are cancelled (async task /
  AbortController; sync: daemon thread, result released when it lands) and an
  earlier-started loser is marked slow (`apply_slow`). All key/engine state
  changes happen on the caller's thread/task — workers only do HTTP. A stream
  races to its first emittable item (`_open`/`openStream`), then the winner's
  generator continues on the caller. Attempts still running at the deadline are
  recorded as `Transient` timeouts (so the key cools). Default strategy is
  `smart`: (priority, `expected_latency`) where an unmeasured/stale (>10 min)
  provider counts as `LATENCY_PRIOR_MS`.
- **Error taxonomy** (`classify()`) — *no status aborts a call on its own*:
  401/403 (and 400 "API key not valid"/location errors) `AuthError` and 402
  (or a 429 saying the quota is gone "/ month") `QuotaExhausted` → disable
  key, fail over (a 402 naming the model benches the model instead; a 403
  saying the model needs a paid plan is `ModelNotFound(gone=True)`); 429
  `RateLimited` → cool key (an hour when the body says the *daily* quota is
  spent and no retry time is given — `DAILY_QUOTA_RETRY`), or if model-scoped per
  `rate_limit_scope` (Gemini, Groq, Z.ai, Cohere: per-model quotas) bench just the model;
  408/409/425/any 5xx `Transient` → breaker + backoff, or if model-scoped per
  `transient_scope` (Gemini "high demand") bench the model; 404/410 and
  400/422 "decommissioned/no longer..." `ModelNotFound(gone=True)` → bench the
  model for an hour (capability 404s like "no endpoints support tool use" are
  `gone=False`); 413 / "context too long" / other 400s mentioning "model" →
  `ModelNotFound(gone=False)`, next model; any other 4xx (incl. OpenRouter
  moderation 403) → `BadRequest`: skip that *provider*, no breaker penalty,
  and raise only once 2 distinct providers rejected the request (or every
  attempt was a rejection). `{"error": ...}` bodies/SSE frames delivered with
  HTTP 200 are classified like the status they carry. Exhaustion raises
  `NoProvidersAvailable(attempts, status)` with one explanation line per
  provider.
- **Model benching** has two levels: provider-wide (`Provider.bench_model`,
  for causes that hit everyone — retired 410/"end of life", OpenRouter
  `rate_limit_scope == "upstream"`, model-scoped overload 5xx) and per key
  (`KeyState.bench_model`, for that account's per-model quota — scope
  `"model"` 429s on Gemini/Groq/Z.ai/Cohere/OVH — "no access" 404s and 402s naming the
  model). Benched candidates keep their rank slot in `order_candidates` and
  are skipped in `select_candidate` — dropping them would let a provider full
  of dead models monopolise rank 0 and starve the interleave. `soonest_wait`
  works over candidates (key wait vs. both bench levels) so `wait=True` also
  waits out per-model quotas.
- **Concrete-id routing**: a non-virtual id goes only to providers whose model
  list contains it; an id nobody lists passes through to all. The free guard
  skips its provider (ConfigError only if nothing else can serve).
- **Virtual models** (`registry`): `auto`/`chat`/`large`/`fast`/`small` plus
  capability tags `tools`/`vision`/`reasoning` (+ `chat:<tag>`); anything whose
  base isn't a known alias passes through verbatim — including ids with `:`
  suffixes like OpenRouter's `:free`. Don't reintroduce fan-out for unknown ids.
  Resolution order = `ModelSpec.priority` (stable), then provider `prefer=`
  patterns; `Provider.resolve_models` also accepts a *list* of aliases (per-call
  fallback chain, deduped in order).
- **Free guard**: providers with `free_only=True` (OpenRouter, Kilo, Z.ai) reject a
  concrete id unless it is `:free` or listed free (`ModelSpec.free`, from
  discovery: `:free`, Kilo `isFree`, or zero pricing). Aliases never resolve to
  a paid model there. Z.ai has no discovery: its explicit list holds only the
  free Flash models, so every other GLM id is refused. Other providers keep
  `free_only=False` (whole account is free-tier).
- **Keyless providers** (`Kilo`, `OVHcloud`; `keyless = True`, placeholder key
  `ANONYMOUS` → no Authorization header, masked as `(keyless)`): the library
  adds them only on request (`keyless=True|"auto"`, `FREELM_KEYLESS`); the CLI
  defaults to `auto` (only when no keys are set) and prints a notice. Never
  route prompts to them silently. OVHcloud never takes a key (paid there).
- **Events**: clients accept `on_event`/`onEvent`; emit kinds
  `attempt|hedge|success|error|wait|discovery`, masked keys only, and swallow callback
  exceptions.
- **Persistence** (`_state.py`/`state.ts`): opt-in (`persist=`/`FREELM_PERSIST`),
  one JSON schema shared by both languages (`provider:sha256(key)[:12]` →
  rpd/cooldown/disabled with wall-clock timestamps). Never write raw keys.
- **CLI** (`_cli.py`/`cli.ts`): stdlib/zero-dep only (argparse /
  `util.parseArgs`); commands `chat|models|health|doctor|serve`; config/usage
  errors exit 2, other freelm errors exit 1. `doctor` sends one tiny chat per
  key (`/models` returns 200 for dead keys on several providers, so it can't be
  trusted) and prints a fix + signup link per failure; the provider table
  (`PROVIDER_ENV`) lives in `config`. `option_vars`/`optionVars` feed extra
  constructor options from the env (Cloudflare's `CLOUDFLARE_ACCOUNT_ID`, which
  is part of its URL) through `build_provider`/`buildProvider` — used by both
  `from_env` and `doctor`; a missing required option skips that provider with a
  stderr notice (doctor: a FAIL row), never a crash.
- **Discovery:** providers with `discover=True` (all except Google, NIM, Z.ai,
  Cohere and Cloudflare; an explicit `models=[...]` turns it off)
  fetch `GET /models` on first use; resolution is live → disk cache
  (`~/.cache/freelm`, TTL 1 h, 0600) → hardcoded `DEFAULT_MODELS` fallback. A
  cached list yielding zero usable specs must fall through to a live fetch.
- **Streaming:** one SSE decoder per language (`_SSE` / `SSE`: comments,
  multi-line `data:`, CR/LF/CRLF, `[DONE]` ends the stream). The core yields
  raw `chat.completion.chunk` dicts (`stream_chunks`/`astream_chunks`/
  `streamChunks`); `stream()` maps them to text. Failover only before the
  first output (raw mode holds role-only preambles back); success records
  time-to-first-token into the latency EWMA. `apply_success` ignores latency
  samples <= 0 — keep it that way. A numeric `delta.content` (Workers AI
  streams some tokens as JSON numbers) is repaired to text before anything
  else sees the chunk (`_repair_chunk`/`repairChunk`).
- **Provider quirks** go in provider hooks, not the client: `rate_limit_scope`/
  `transient_scope` (who a 429/5xx throttles) and `adapt_payload` (last look at
  the request body — Cloudflare adds `max_tokens`, since many Workers AI models
  stop at 256 tokens by default).
- **`serve`**: zero-dep OpenAI-compatible endpoint. Python: stdlib
  `ThreadingHTTPServer`, every request handed to one `AsyncFreeLLM` on a
  private loop thread (single-loop safety). TS: `node:http` loaded via
  `runtime.builtin()`; client disconnects abort upstream through `signal`.
- **TS timeouts:** `fetch` doesn't bound body reads. Non-stream: one timer
  spans headers *and* body, bounded by the call's remaining deadline. Streams:
  a headers timer, then each `reader.read()` raced against an inactivity timer
  that runs **only while waiting on the network** — never while the consumer
  holds a yielded chunk (that hung forever before 0.4). Timers are `unref`'d.
  Discovery uses `AbortSignal.timeout(10_000)`. Caller `signal` is linked into
  every fetch and surfaces as an AbortError without penalising the key.
- **TS runtime portability:** no static `node:` imports in library code —
  `runtime.ts` reads `process.env` and Node built-ins via
  `process.getBuiltinModule`, so the package bundles for Workers/edge/browsers
  (disk cache + persistence silently off there). Only `cli.ts` imports `node:`
  modules directly.

## Conventions

- Dependencies: Python runtime dep is httpx only; TS has **zero** runtime deps.
  Don't add any without strong reason.
- Lint: ruff's rule set is pinned in `pyproject.toml` (`select = E4,E7,E9,F,I`)
  — ruff 0.16 widened its defaults and broke CI without a code change.
- Free-only policy: paid-only providers are out of scope (xAI Grok explicitly
  rejected; Groq `gsk_…` is the supported one).
- Secrets: keys come from env (see `.env.example`); never logged or committed.
  `KeyState.masked()` for any key display. Cache files are chmod 0600.
- Versions are single-sourced: `src/freelm/_version.py` (hatchling reads it;
  UA derives from it) and `js/src/version.ts` (must match `js/package.json`,
  enforced by a vitest case). Bump version + CHANGELOG.md together.
- Provider tier limits (rpm/rpd) are dated heuristics — when touching them,
  update the "verified YYYY-MM" comments.
- Tests: respx (Python) / `vi.stubGlobal("fetch", ...)` (TS). No real network
  in unit tests. Discovery tests must isolate `FREELM_CACHE_DIR` to a tmp dir.
- Concurrency: sync `FreeLLM` mutates key state without locks (documented:
  one client per thread); the async clients are single-event-loop safe.

## Release

- Versions are aligned since 0.4.0 (Python and npm ship the same number).
- Python: bump `_version.py`, update CHANGELOG, publish a GitHub Release
  (tag `v*`) → `release.yml` tests then uploads to PyPI. Auth: PyPI Trusted
  Publishing (OIDC, environment `pypi`) once configured on pypi.org; until
  then the `PYPI_API_TOKEN` secret.
- JS: bump `js/package.json` **and** `js/src/version.ts`, push tag `js-v*`
  → `npm-release.yml` builds, tests, publishes with provenance. Auth: npm
  Trusted Publishing (OIDC, npm >= 11.5.1) once configured on npmjs.com; the
  `NPM_TOKEN` fallback must be rotated (npm write tokens expire <= 90 days).
- Docker: `docker.yml` builds `ghcr.io/shihabshahrier/freelm` (`freelm serve`,
  amd64+arm64) on `v*` tags with the built-in `GITHUB_TOKEN`.
- CI: `ci.yml` (Python 3.9–3.14, ruff + pytest), `js-ci.yml` (Node 20/22/24,
  tsc + vitest + build + ESM/CJS/CLI smoke; path-filtered to `js/**`).
