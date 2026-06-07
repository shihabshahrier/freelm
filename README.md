# freelm — free, always-up LLM client for Python

[![PyPI version](https://img.shields.io/pypi/v/freelm.svg)](https://pypi.org/project/freelm/)
[![Python versions](https://img.shields.io/pypi/pyversions/freelm.svg)](https://pypi.org/project/freelm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**freelm is a free, always-up LLM client and gateway for Python** that pools multiple free-tier LLM providers — **OpenRouter, Google Gemini (AI Studio), and NVIDIA NIM** — behind one OpenAI-compatible call, with automatic API-key rotation, cross-provider failover, circuit breaking, rate-limit/quota-aware routing, and live free-model discovery. Drop in whichever free keys you have and your app keeps talking to an LLM even when one source rate-limits or goes down.

📦 **PyPI:** https://pypi.org/project/freelm/ — `pip install freelm`

> Python first. JS/TS and Go ports planned (the core is spec-driven for portability).

## Why

LLMs show up in nearly every project, and they cost money — but there's a lot of *free* capacity scattered across providers:

- **OpenRouter** — free models (`:free`), ~50 req/day under $10 credit, ~1000/day at ≥$10.
- **Google AI Studio (Gemini)** — generous free tier; Tier 1 (billing on) lifts limits hard.
- **NVIDIA NIM** (`build.nvidia.com`) — many models free against build credits.

`freelm` pools them behind one fault-tolerant client.

## Install

```bash
pip install freelm
```

## Quick start

```python
import freelm

llm = freelm.FreeLLM.from_env()          # reads keys from environment
print(llm.text("Explain black holes in one sentence."))
```

Explicit config:

```python
from freelm import FreeLLM, OpenRouter, GoogleAIStudio, NIM

llm = FreeLLM(
    providers=[
        OpenRouter("sk-or-...", tier="free"),       # or tier="credit" if ≥ $10
        GoogleAIStudio("AIza...", tier="free"),      # or tier="tier1"
        NIM("nvapi-..."),
    ],
    strategy="quota_aware",   # priority | round_robin | quota_aware | latency
)

resp = llm.chat(
    [{"role": "user", "content": "Write a haiku about failover."}],
    model="chat:fast",        # virtual model, see below
)
print(resp.text, "via", resp.provider)
```

Async is symmetric:

```python
from freelm import AsyncFreeLLM

async with AsyncFreeLLM.from_env() as llm:
    print(await llm.text("hi"))
```

## Drop-in OpenAI shim

```python
# from openai import OpenAI
from freelm.compat import OpenAI

client = OpenAI()                          # backed by FreeLLM.from_env()
r = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "hi"}],
)
print(r.choices[0].message.content)
```

## Environment variables

| Provider | Key vars (first match wins) | Tier var |
|----------|------------------------------|----------|
| OpenRouter | `OPENROUTER_API_KEY` / `FREELM_OPENROUTER_KEYS` | `FREELM_OPENROUTER_TIER` (`free`\|`credit`) |
| Google AI Studio | `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `GOOGLE_AI_STUDIO_KEY` / `FREELM_GOOGLE_KEYS` | `FREELM_GOOGLE_TIER` (`free`\|`tier1`) |
| NVIDIA NIM | `NVIDIA_API_KEY` / `NIM_API_KEY` / `FREELM_NIM_KEYS` | `FREELM_NIM_TIER` (`free`) |

Multiple keys per provider: comma-separate them.

## Virtual models

Names differ per provider, so ask by intent and `freelm` maps to a concrete model:

| Alias | Meaning |
|-------|---------|
| `auto` / `chat` | any available chat model (registry order) |
| `chat:large` / `large` | a larger/stronger model |
| `chat:fast` / `fast` | a fast/cheap model |
| `chat:small` / `small` | smallest model |
| `vendor/model-id` | passthrough — use exactly this model |

Override the table per provider with `models=[ModelSpec(...)]`.

## Dynamic model discovery

Free model IDs churn constantly, so `freelm` **doesn't trust its hardcoded list**. For OpenRouter (on by default), it queries `GET /models` on first use, derives tags (`large`/`fast`/`small`, plus `tools`/`vision`/`reasoning` from `supported_parameters`), and caches the list to disk.

Resolution order: **live API → disk cache → hardcoded fallback** (so it still works offline / key-less).

```python
from freelm import list_free_models

for m in list_free_models()[:5]:        # live OpenRouter free models, cached
    print(m.id, m.tags, m.ctx)
```

Control it:

```python
OpenRouter("sk-or-...", discover=True, discover_free_only=True, cache_ttl=3600)
GoogleAIStudio("AIza...", discover=True)   # opt-in for other providers' /models

llm.refresh_models()                        # force re-fetch on next call
```

| Env var | Default | Meaning |
|---------|---------|---------|
| `FREELM_CACHE_DIR` | `~/.cache/freelm` | where the model cache lives (file is `0600`) |
| `FREELM_CACHE_TTL` | `3600` | cache lifetime in seconds |

## Configuration & tuning

Client knobs — `FreeLLM(...)` / `AsyncFreeLLM(...)`:

| Param | Default | What it does |
|-------|---------|--------------|
| `strategy` | `"priority"` | how providers are ranked (see below) |
| `max_attempts` | `12` | hard cap on total tries across all providers/keys/models per call |
| `timeout` | `60.0` | per-request timeout (s); also the overall deadline for one `chat()` |
| `wait` | `False` | if every key is cooling, sleep until one frees instead of failing |
| `max_wait` | `20.0` | longest single sleep (s) when `wait=True` |
| `http_client` | `None` | bring your own `httpx.Client` / `AsyncClient` |

Provider knobs — `OpenRouter(...)`, `GoogleAIStudio(...)`, `NIM(...)`:

| Param | Default | What it does |
|-------|---------|--------------|
| `keys` | — | one key (str) or many (list, or comma-string via env) |
| `tier` | `"free"` | selects built-in rpm/rpd limits |
| `priority` | `0` | **lower = tried first** (with `strategy="priority"`) |
| `rpm` / `rpd` | tier default | override requests-per-minute / per-day |
| `models` | discovered / built-in | override model list (order = preference) |
| `discover` | OpenRouter `True`, else `False` | live-fetch `/models` |
| `cache_ttl` | env / 1h | discovery cache lifetime |

### Strategies

| Strategy | Behaviour |
|----------|-----------|
| `priority` | providers in ascending `priority`, then list order. Deterministic. |
| `round_robin` | rotate which provider goes first each call. Spreads load evenly. |
| `quota_aware` | rank by current headroom (rpm tokens bounded by daily quota); cooling/disabled keys score 0. Unlimited-quota providers rank high but **deplete as used**, so traffic still spreads. |
| `latency` | prefer the provider with the lowest observed average latency. |

Whatever the ranking, candidates are **interleaved across providers** — the best model of *every* provider is tried before any provider's 2nd model — so failover always reaches every provider, even when your first provider has dozens of throttled free models.

### Defining your own priority order

```python
from freelm import FreeLLM, OpenRouter, GoogleAIStudio, NIM

llm = FreeLLM(
    [
        OpenRouter("sk-or-...",   priority=0),   # try first
        GoogleAIStudio("AIza...", priority=1),   # then this
        NIM("nvapi-...",          priority=2),   # last resort
    ],
    strategy="priority",
)
```

Within a provider, model preference is the order of its `models` list:

```python
from freelm import OpenRouter, ModelSpec

OpenRouter("sk-or-...", discover=False, models=[
    ModelSpec("openai/gpt-oss-120b:free", ("chat", "large")),
    ModelSpec("meta-llama/llama-3.3-70b-instruct:free", ("chat", "large")),
])
```

## Errors

```python
from freelm import NoProvidersAvailable, ProviderError

try:
    resp = llm.chat("hi")
except NoProvidersAvailable as e:
    print("all providers exhausted:", e.attempts)   # [(candidate, exception), ...]
except ProviderError as e:
    print(e.provider, e.status, e.retryable)         # e.g. a malformed 400
```

Hierarchy: `FreeLLMError` → `ConfigError` · `NoProvidersAvailable` · `ProviderError` → `AuthError` / `RateLimited` / `Transient` / `ModelNotFound`. Retryable errors (`RateLimited`, `Transient`) are handled internally and only surface, bundled, inside `NoProvidersAvailable`.

## Response & introspection

```python
r = llm.chat("hi")
r.text          # assistant text (also: str(r))
r.provider      # which provider served it, e.g. "openrouter"
r.model         # concrete model id used
r.usage         # .prompt_tokens / .completion_tokens / .total_tokens
r.latency_ms    # round-trip latency
r.raw           # original provider JSON
```

`llm.health()` → one dict per key: `provider`, `key` (masked), `ready`, `breaker`, `rpd_used`, `last_error`, `ewma_latency_ms`.

> **Concurrency:** `AsyncFreeLLM` is safe across many concurrent tasks on one event loop. A sync `FreeLLM` mutates per-key state without locks — use one client per thread, or use the async client, for multi-threaded workloads.

## How "always-up" works

- **Key pool** per provider, round-robined to spread load.
- **Failover chain**: interleaved across providers (best model of each, then next-best) so every provider is reached fast — never starved by one provider's many models.
- **Circuit breaker** per key: opens after repeated failures, half-opens after a cooldown — no hammering a dead key.
- **Retry classification**: `429` → cool the key & rotate; `5xx`/timeout → breaker + backoff; `401/403` → disable the key; `4xx` model errors → try another model/provider; other `4xx` → surfaced as a caller bug.
- **Quota guard**: per-key requests/minute (token bucket) + requests/day counter, so a key predicted to be exhausted is skipped before you waste a call.
- **`wait=True`** (optional): briefly sleep until a key frees up instead of failing, bounded by `max_wait`.

Inspect live state any time:

```python
for row in llm.health():
    print(row)   # provider, key (masked), ready, breaker, rpd_used, last_error, latency
```

## Roadmap

- v1.1 — streaming (SSE normalization across providers)
- v1.2 — persistent quota tracking (sqlite/json) + tighter tier pacing
- v1.3 — tool / function-calling normalization
- v2 — embeddings, vision; JS/TS and Go ports

## FAQ

### How do I use free LLMs in Python?
Install `freelm`, set one or more free API keys (OpenRouter, Google AI Studio, or NVIDIA NIM) as environment variables, and call `freelm.FreeLLM.from_env().text("...")`. freelm picks an available free model and handles rate limits and failover automatically.

### How do I fall back between OpenRouter, Gemini, and NVIDIA NIM?
Pass several providers to `FreeLLM([...])`. On a rate limit (`429`), dead key (`401`), or server error, freelm rotates keys and fails over to the next provider — interleaved so every provider is reached quickly instead of stalling on one.

### Is there an OpenAI-compatible free LLM client?
Yes — `from freelm.compat import OpenAI` is a drop-in for the OpenAI SDK (`client.chat.completions.create(...)`), backed by free providers.

### How do I avoid free-tier rate limits?
freelm paces each key with a requests-per-minute token bucket plus a daily counter and skips keys predicted to be exhausted. Add more keys or providers to raise total throughput.

### Which free LLM models are available right now?
Free model IDs change constantly, so freelm discovers them live from the provider API and caches them. Run `from freelm import list_free_models; list_free_models()` for the current list.

### Is freelm really free?
freelm itself is MIT-licensed and free. It runs on providers' free tiers; the actual request limits depend on each provider's free quota.

## License

MIT © Shahriar Labs

> Free-tier model lists change often — `freelm` discovers OpenRouter models live and caches them, so you rarely touch the hardcoded list. Tier **rate-limit numbers** are still heuristic defaults; override `rpm`/`rpd`/`tier` as providers evolve.
