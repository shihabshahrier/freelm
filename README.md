# freelm — free, always-up LLM client for Python & JavaScript

[![PyPI version](https://img.shields.io/pypi/v/freelm.svg)](https://pypi.org/project/freelm/)
[![Python versions](https://img.shields.io/pypi/pyversions/freelm.svg)](https://pypi.org/project/freelm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**freelm is a free, always-up LLM client and gateway for Python (and JS/TS)** that pools six free-tier LLM providers — **OpenRouter, Google Gemini (AI Studio), NVIDIA NIM, Groq, Cerebras, and Mistral** — behind one OpenAI-compatible call (with streaming), with automatic API-key rotation, cross-provider failover, circuit breaking, rate-limit/quota-aware routing, and live free-model discovery. Drop in whichever free keys you have and your app keeps talking to an LLM even when one source rate-limits or goes down.

📦 **PyPI:** https://pypi.org/project/freelm/ — `pip install freelm`

🌐 **Website & docs:** https://shihub.online/freelm · https://shihub.online/freelm/docs

> Python + JS/TS (`npm install freelm`, lives in [`js/`](js/)). A Go port is planned (the core is spec-driven for portability).

## Why

LLMs show up in nearly every project, and they cost money — but there's a lot of *free* capacity scattered across providers:

- **OpenRouter** — free models (`:free`), ~50 req/day under $10 credit, ~1000/day at ≥$10.
- **Google AI Studio (Gemini)** — generous free tier; Tier 1 (billing on) lifts limits hard.
- **NVIDIA NIM** (`build.nvidia.com`) — many models free against build credits.
- **Groq** — 30 RPM / 14,400 req-day free, very fast inference, no card.
- **Cerebras** — ~30 RPM, **1M tokens/day** free (8K context cap), no card.
- **Mistral** — free "Experiment" tier: 2 RPM, 500K TPM, 1B tokens/month.

`freelm` pools them behind one fault-tolerant client.

> Free-tier numbers above were verified 2026-06 and change often — they're defaults you can override with `tier` / `rpm` / `rpd`.

## Install

```bash
pip install freelm
```

**JavaScript / TypeScript:** `npm install freelm` — same API, in [`js/`](js/). Zero runtime deps (built-in `fetch`).

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

## Streaming

Token streaming works across every provider and through the same failover. It fails over between providers **before the first token**; once tokens start flowing it stays on that provider (no mid-stream switching).

```python
llm = freelm.FreeLLM.from_env()
for chunk in llm.stream("Write a haiku about failover."):
    print(chunk, end="", flush=True)
```

```python
async with freelm.AsyncFreeLLM.from_env() as llm:
    async for chunk in llm.astream("Stream me some tokens"):
        print(chunk, end="", flush=True)
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

OpenAI-SDK constructor arguments (`api_key`, `base_url`, `organization`, ...) are
accepted and ignored — keys come from the environment. `stream=True` works and
yields `chat.completion.chunk`-shaped objects:

```python
for chunk in client.chat.completions.create(model="auto", messages=msgs, stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

## Environment variables

| Provider | Key vars (first match wins) | Tier var |
|----------|------------------------------|----------|
| OpenRouter | `OPENROUTER_API_KEY` / `FREELM_OPENROUTER_KEYS` | `FREELM_OPENROUTER_TIER` (`free`\|`credit`) |
| Google AI Studio | `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `GOOGLE_AI_STUDIO_KEY` / `FREELM_GOOGLE_KEYS` | `FREELM_GOOGLE_TIER` (`free`\|`tier1`) |
| NVIDIA NIM | `NVIDIA_API_KEY` / `NIM_API_KEY` / `FREELM_NIM_KEYS` | `FREELM_NIM_TIER` (`free`) |
| Groq | `GROQ_API_KEY` / `FREELM_GROQ_KEYS` | `FREELM_GROQ_TIER` (`free`) |
| Cerebras | `CEREBRAS_API_KEY` / `FREELM_CEREBRAS_KEYS` | `FREELM_CEREBRAS_TIER` (`free`) |
| Mistral | `MISTRAL_API_KEY` / `FREELM_MISTRAL_KEYS` | `FREELM_MISTRAL_TIER` (`free`) |

Multiple keys per provider: comma-separate them. See `.env.example`.

> **Groq vs xAI Grok:** different companies. **Groq** (`gsk_…`) has a free tier and is supported. **xAI Grok** (`xai-…`) is paid, so it's intentionally *not* included — freelm is free-only.

## Virtual models

Names differ per provider, so ask by intent and `freelm` maps to a concrete model:

| Alias | Meaning |
|-------|---------|
| `auto` / `chat` | any available chat model (priority, then registry order) |
| `chat:large` / `large` | a larger/stronger model |
| `chat:fast` / `fast` | a fast/cheap model |
| `chat:small` / `small` | smallest model |
| `chat:tools` / `tools` | models that support function calling |
| `vision` / `reasoning` | models tagged with that capability |
| `vendor/model-id` | passthrough — use exactly this model |

Override the table per provider with `models=[ModelSpec(...)]`.

## Model & provider priority

Three ways to control *which model wins*, from static to per-call:

```python
from freelm import FreeLLM, OpenRouter, ModelSpec

# 1. ModelSpec(priority=) — order a static list (lower = first)
OpenRouter("sk-or-...", discover=False, models=[
    ModelSpec("openai/gpt-oss-120b:free", ("chat", "large"), priority=1),
    ModelSpec("meta-llama/llama-3.3-70b-instruct:free", ("chat", "large"), priority=0),
])

# 2. prefer=[...] — bias *discovered* lists without replacing them
#    (exact id, else case-insensitive substring; survives refresh_models())
OpenRouter("sk-or-...", prefer=["qwen/qwen3-next-80b-a3b-instruct:free", "gpt-oss"])

# 3. per-call ordered fallback chain — ids and aliases mix freely
llm.chat(msgs, model=["groq-only/llama-3.3-70b-versatile", "chat:fast"])
```

Provider `priority=` (lower = tried first) is now the universal tiebreak: primary
for `strategy="priority"`, secondary for `quota_aware`/`latency` (equal headroom
or latency → lower priority wins), and the baseline order for `round_robin`.

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
| `on_event` | `None` | observability callback — see below |
| `persist` | `False` / `FREELM_PERSIST` | carry quota/cooldown/disabled state across restarts |
| `http_client` | `None` | bring your own `httpx.Client` / `AsyncClient` |

Provider knobs — `OpenRouter(...)`, `GoogleAIStudio(...)`, `NIM(...)`:

| Param | Default | What it does |
|-------|---------|--------------|
| `keys` | — | one key (str) or many (list, or comma-string via env) |
| `tier` | `"free"` | selects built-in rpm/rpd limits |
| `priority` | `0` | **lower = tried first** (tiebreak in every strategy) |
| `prefer` | `[]` | model ids/substrings to move to the front of resolution |
| `free_only` | OpenRouter `True`, else `False` | block paid model ids (see below) |
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

## When can freelm cost money?

freelm is free-only **by default and by guard**, not just by convention:

- **OpenRouter** mixes paid and free models in one catalog, so it ships with
  `free_only=True`: passing a non-`:free` model id raises `ConfigError` instead
  of silently billing you. Opt out per provider: `OpenRouter(key, free_only=False)`.
- **Google AI Studio** is free unless *you* pick `tier="tier1"` (billing enabled).
- **NVIDIA NIM** burns build.nvidia.com credits — free until they run out (requests then fail, not bill).
- **Groq / Cerebras / Mistral** free-tier accounts: every model is free at that tier.

## Tool calling & JSON output

`tools`, `tool_choice`, and `response_format` pass straight through to the
provider; `chat:tools` routes to models that support function calling:

```python
r = llm.chat(msgs, model="chat:tools", tools=[...], tool_choice="auto")
r.tool_calls                 # [{"id": ..., "function": {...}}] or None

llm.chat(msgs, response_format={"type": "json_object"})
```

(Tool calls are non-streaming for now; `stream()` yields text deltas only.)

> **Thinking-model gotcha:** reasoning models (gemini-2.5-flash, gpt-oss, ...)
> can spend a small `max_tokens` budget entirely on hidden reasoning and return
> *empty* text with `finish_reason="length"`. Give them headroom (≥128) or pick
> a non-thinking model — `auto` already deprioritizes `reasoning`-tagged models.

## Observability

Watch every attempt, failover, and success without wrapping the client:

```python
def hook(e):  # freelm.Event
    print(e.kind, e.provider, e.model, e.status, e.latency_ms)

llm = freelm.FreeLLM.from_env(on_event=hook)
# attempt openrouter openai/gpt-oss-20b:free None None
# error   openrouter openai/gpt-oss-20b:free 429 None
# attempt google gemini-2.5-flash None None
# success google gemini-2.5-flash None 412.3
```

`kind` is `attempt | success | error | wait | discovery`; keys are always
masked. A raising callback never breaks the call. `llm.health()` still gives
point-in-time state.

## Persistent quota state

By default counters live in memory, so a restarted process re-burns exhausted
keys. Opt in to disk persistence (shared schema with the JS package):

```python
llm = freelm.FreeLLM.from_env(persist=True)   # or env FREELM_PERSIST=1
```

State (`rpd_used`, cooldowns, disabled keys — never raw keys, only hashes)
lives in `~/.cache/freelm/state.json` (0600), loaded at construction and saved
after each call. Multi-process is last-writer-wins, best effort.

## CLI

The package installs a `freelm` command (`pipx install freelm`, or `npx freelm`
for the JS package):

```bash
freelm chat "explain failover in one line" --model chat:fast --stream
freelm models --provider openrouter     # live free-model list
freelm health                           # per-key readiness/quota table
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

Hierarchy: `FreeLLMError` → `ConfigError` · `NoProvidersAvailable` · `ProviderError` → `AuthError` / `QuotaExhausted` / `RateLimited` / `Transient` / `ModelNotFound`. Retryable errors (`RateLimited`, `Transient`) are handled internally and only surface, bundled, inside `NoProvidersAvailable`. `AuthError` (401/403) and `QuotaExhausted` (402, e.g. OpenRouter out of credits) disable the key and fail over instead of aborting the call.

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
- **Retry classification**: `429` → cool the key & rotate; `5xx`/timeout → breaker + backoff; `401/403`/`402` → disable the key; `4xx` model errors → try another model/provider; other `4xx` → surfaced as a caller bug.
- **Quota guard**: per-key requests/minute (token bucket) + requests/day counter, so a key predicted to be exhausted is skipped before you waste a call.
- **`wait=True`** (optional): briefly sleep until a key frees up instead of failing, bounded by `max_wait`.

Inspect live state any time:

```python
for row in llm.health():
    print(row)   # provider, key (masked), ready, breaker, rpd_used, last_error, latency
```

## Roadmap

Shipped: streaming (0.2.0), JS/TS port (npm `freelm`), model/provider priority +
free-only guard + tool-calling passthrough + observability + CLI + persistent
quota state (0.3.0).

- next — token-based pacing (TPM/TPD budgets from response usage)
- then — streaming tool calls; deeper structured-output normalization
- later — embeddings, vision; Go port

## How freelm compares

freelm is a **client-side, free-tier-only failover layer** — not a proxy server, not an agent framework:

| Tool | What it is | How freelm differs |
|------|------------|--------------------|
| **LiteLLM** | SDK + proxy server for 100+ providers (paid & free) | freelm is free-only, zero-infrastructure (no proxy to run), with quota/breaker state per key built in |
| **OpenRouter SDK** | Client for one aggregator | OpenRouter is *one* of freelm's six pools — when its free quota dries up, freelm fails over to Gemini, Groq, Cerebras, Mistral, or NIM directly |
| **LangChain / LlamaIndex** | Orchestration frameworks | freelm is a thin client; use it *inside* them via the OpenAI-compatible shim |

## FAQ

### How do I use free LLMs in Python?
Install `freelm`, set one or more free API keys (OpenRouter, Google AI Studio, NVIDIA NIM, Groq, Cerebras, or Mistral) as environment variables, and call `freelm.FreeLLM.from_env().text("...")`. freelm picks an available free model and handles rate limits and failover automatically.

### How do I use free LLMs in JavaScript / Node.js / TypeScript?
`npm install freelm` (Node ≥ 18, zero runtime dependencies), set the same env keys, and call `await FreeLLM.fromEnv().text("...")`. The TypeScript package mirrors the Python API — same providers, same failover engine, same streaming.

### Is there a free alternative to the OpenAI API?
Yes. Six providers ship usable free tiers in 2026 — OpenRouter (`:free` models), Google AI Studio, NVIDIA NIM, Groq, Cerebras, and Mistral — and `freelm.compat.OpenAI` is a drop-in for the OpenAI SDK that routes `chat.completions.create(...)` across all of them with automatic failover, including `stream=True`.

### Which LLM providers have free API tiers in 2026?
Verified 2026-06: **OpenRouter** (~50 req/day under $10 lifetime credit, ~1000/day at ≥$10), **Google AI Studio** (per-model free quotas), **NVIDIA NIM** (free against build.nvidia.com credits), **Groq** (30 RPM / 14,400 req/day, no card), **Cerebras** (~30 RPM, 1M tokens/day), **Mistral** (Experiment tier: 2 RPM, 1B tokens/month). freelm ships these as tier defaults you can override.

### How do I fall back between OpenRouter, Gemini, Groq, and the other free providers?
Pass several providers to `FreeLLM([...])`. On a rate limit (`429`), dead key (`401`), exhausted credits (`402`), or server error, freelm rotates keys and fails over to the next provider — interleaved so every provider is reached quickly instead of stalling on one.

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
