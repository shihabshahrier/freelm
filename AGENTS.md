# freelm — agent guide

Free, always-up LLM client + gateway pooling free-tier providers (OpenRouter,
Google AI Studio, NVIDIA NIM, Groq, Cerebras, Mistral) behind one
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
| `_version.py`           | `version.ts`           |

## Commands

```bash
# Python (venv at .venv, Python >= 3.9)
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q                      # tests
.venv/bin/ruff check src tests examples  # lint (CI-enforced)

# TypeScript (Node >= 18)
cd js && npm ci
npm test            # vitest
npm run typecheck   # tsc --noEmit
npm run build       # tsup -> dist/ (esm+cjs+dts)
```

Live smoke test (needs real keys in env, never hardcoded):
`examples/e2e_smoke.py` / `js/examples/e2e.mjs`.

## Architecture (both languages)

- **Layering:** `engine` + `strategy` are pure (no I/O, time injected as
  monotonic seconds) — all orchestration decisions live there so the sync and
  async Python clients and the TS client share identical logic. Only
  `client.*` performs HTTP (httpx in Python, built-in `fetch` in TS).
- **Candidate loop:** a candidate is one (provider, key, model). Per call:
  order candidates by strategy, interleave breadth-first across providers
  (rank 0 of every provider before any rank 1), skip `tried`, `reserve()` a
  token, fire, classify the outcome, repeat up to `max_attempts` within a
  `timeout` deadline.
- **Error taxonomy** (`classify()`): 401/403 `AuthError` and 402
  `QuotaExhausted` → disable key, fail over; 429 `RateLimited` → cool key
  (or, if model-scoped per `rate_limit_scope`, keep key hot and try the next
  model); 408/5xx `Transient` → breaker + backoff; 404 or 400/422 mentioning
  "model" `ModelNotFound` → refund the daily slot, next model; any other 4xx
  → caller bug, raise immediately. Exhaustion raises `NoProvidersAvailable`
  with the attempt list.
- **Virtual models** (`registry`): `auto`/`chat`/`large`/`fast`/`small` plus
  capability tags `tools`/`vision`/`reasoning` (+ `chat:<tag>`); anything whose
  base isn't a known alias passes through verbatim — including ids with `:`
  suffixes like OpenRouter's `:free`. Don't reintroduce fan-out for unknown ids.
  Resolution order = `ModelSpec.priority` (stable), then provider `prefer=`
  patterns; `Provider.resolve_models` also accepts a *list* of aliases (per-call
  fallback chain, deduped in order).
- **Free guard**: OpenRouter defaults `free_only=True` and raises `ConfigError`
  from its `_check_free` hook for non-`:free` passthrough ids. Other providers
  keep the hook a no-op (their whole account is free-tier).
- **Events**: clients accept `on_event`/`onEvent`; emit kinds
  `attempt|success|error|wait|discovery`, masked keys only, and swallow callback
  exceptions.
- **Persistence** (`_state.py`/`state.ts`): opt-in (`persist=`/`FREELM_PERSIST`),
  one JSON schema shared by both languages (`provider:sha256(key)[:12]` →
  rpd/cooldown/disabled with wall-clock timestamps). Never write raw keys.
- **CLI** (`_cli.py`/`cli.ts`): stdlib/zero-dep only; commands
  `chat|models|health`; config errors exit 2, other freelm errors exit 1.
- **Discovery:** providers with `discover=True` (all except Google and NIM)
  fetch `GET /models` on first use; resolution is live → disk cache
  (`~/.cache/freelm`, TTL 1 h, 0600) → hardcoded `DEFAULT_MODELS` fallback. A
  cached list yielding zero usable specs must fall through to a live fetch.
- **Streaming:** SSE deltas; failover only before the first token; success
  records time-to-first-token into the latency EWMA. `apply_success` ignores
  latency samples <= 0 — keep it that way.
- **TS timeouts:** `fetch` doesn't bound body reads, so `client.ts` re-arms an
  AbortController per phase (headers / body / each stream chunk) and discovery
  uses `AbortSignal.timeout(15_000)`. Don't simplify back to a single timer
  cleared after the headers.

## Conventions

- Dependencies: Python runtime dep is httpx only; TS has **zero** runtime deps.
  Don't add any without strong reason.
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

- Python: bump `_version.py`, update CHANGELOG, publish a GitHub Release
  (tag `v*`) → `release.yml` tests then uploads to PyPI (`PYPI_API_TOKEN`).
- JS: bump `js/package.json` **and** `js/src/version.ts`, push tag `js-v*`
  → `npm-release.yml` builds, tests, publishes with provenance (`NPM_TOKEN`).
- CI: `ci.yml` (Python 3.9–3.14, ruff + pytest), `js-ci.yml` (Node 18/20/22,
  tsc + vitest + build; path-filtered to `js/**`).
