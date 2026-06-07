# freelm

**One always-up LLM client over free-tier providers.** Drop in your OpenRouter, Google AI Studio, and/or NVIDIA NIM keys, and `freelm` gives you a single chat call that auto-rotates keys, fails over across providers, paces itself to each tier's limits, and trips circuit breakers on dead keys — so your app keeps talking to an LLM even when one source rate-limits or dies.

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

## How "always-up" works

- **Key pool** per provider, round-robined to spread load.
- **Failover chain**: key → next key → next provider until one succeeds.
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

## License

MIT © Shahriar Labs

> Free-tier model lists change often — `freelm` discovers OpenRouter models live and caches them, so you rarely touch the hardcoded list. Tier **rate-limit numbers** are still heuristic defaults; override `rpm`/`rpd`/`tier` as providers evolve.
