# freelm — free LLM API for Python & Node.js (OpenAI-compatible, auto-failover)

[![PyPI](https://img.shields.io/pypi/v/freelm?label=PyPI)](https://pypi.org/project/freelm/)
[![npm](https://img.shields.io/npm/v/freelm?label=npm)](https://www.npmjs.com/package/freelm)
[![CI](https://github.com/shihabshahrier/freelm/actions/workflows/ci.yml/badge.svg)](https://github.com/shihabshahrier/freelm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/shihabshahrier/freelm/blob/main/LICENSE)

**freelm turns the free tiers of Google Gemini, Groq, OpenRouter, Cloudflare Workers AI, Z.ai, Cohere, Mistral,
NVIDIA NIM and Kilo into one OpenAI-compatible LLM — in your Python or TypeScript code, or as a local `/v1` endpoint for any tool.** It rotates
your keys, fails over across providers on rate limits, outages and retired models, and discovers which models are
free today. Your own free keys, called directly: nothing to host, no relay in the middle — and the CLI even works
with **no keys at all**.

```bash
pip install freelm          # Python >= 3.9   ·   npm install freelm  (Node >= 20, zero dependencies)
```

```python
import freelm

llm = freelm.FreeLLM.from_env()      # reads GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, ...
print(llm.text("Explain failover in one sentence."))
```

## Start in 60 seconds

**Zero keys?** `pip install freelm && freelm chat "hello"` already works: with no keys configured, the CLI falls
back to keyless public endpoints ([Kilo Gateway](https://kilo.ai) free routes, [OVHcloud AI Endpoints](https://endpoints.ai.cloud.ovh.net)
anonymous tier) and says so. Their limits are low and free routes may log prompts, so for real use:

1. **Get one free key** (no credit card): [Google AI Studio](https://aistudio.google.com/apikey) is the most
   generous; add [Groq](https://console.groq.com/keys) and [OpenRouter](https://openrouter.ai/keys) for failover.
2. `export GEMINI_API_KEY=...` (and any others — see [providers](#free-providers)).
3. **Check what works:** `freelm doctor` sends one tiny request per key and tells you exactly what's wrong and how
   to fix it:

   ```text
   $ freelm doctor
     openrouter  sk-or-...a1b2    FAIL  key rejected (401): User not found. — get a new free key: https://openrouter.ai/keys
     google      AIzaSy...x9yz    OK    gemini-2.5-flash-lite · 1131 ms
     groq        gsk_ab...cd12    FAIL  key rejected (401): Invalid API Key — get a new free key: https://console.groq.com/keys
   1 of 3 key(s) working — ready: google
   ```
4. Call it from code (above), the terminal (`freelm chat "hi"`), or any OpenAI-compatible tool (`freelm serve`, below).

## Use free LLMs in any OpenAI-compatible tool

`freelm serve` runs the same router as a local OpenAI-compatible API — streaming, tool calls and failover included:

```bash
freelm serve                 # → http://127.0.0.1:4000/v1   (npx freelm serve works too)
```

Then point your tool at `http://127.0.0.1:4000/v1` with any API key and model `auto` (or `chat:fast`, `large`,
`tools`, or a concrete id):

| Tool | Setting |
|------|---------|
| **OpenAI SDK** (Python / Node) | `OpenAI(base_url="http://127.0.0.1:4000/v1", api_key="freelm")` |
| **LangChain** | `ChatOpenAI(base_url="http://127.0.0.1:4000/v1", api_key="freelm", model="auto")` |
| **LlamaIndex** | `OpenAILike(api_base="http://127.0.0.1:4000/v1", api_key="freelm", model="auto", is_chat_model=True)` |
| **Vercel AI SDK** | `createOpenAICompatible({ name: "freelm", baseURL: "http://127.0.0.1:4000/v1" })("auto")` |
| **Continue** (VS Code / JetBrains) | model with `provider: openai`, `apiBase: http://127.0.0.1:4000/v1`, `model: auto` |
| **Cline / Roo Code** | API provider "OpenAI Compatible", base URL `http://127.0.0.1:4000/v1`, model `auto` |
| **OpenCode** | `opencode.json` → `"provider": {"freelm": {"npm": "@ai-sdk/openai-compatible", "options": {"baseURL": "http://127.0.0.1:4000/v1"}, "models": {"auto": {}}}}` ([docs](https://opencode.ai/docs/providers/)) |
| **OpenClaw** | `models.providers.freelm`: `baseUrl: "http://127.0.0.1:4000/v1"`, `api: "openai-completions"`, `models: [{ id: "auto" }]`, then allow `freelm/auto` in `agents.defaults.models` ([docs](https://docs.openclaw.ai/gateway/config-tools/custom-providers)) |
| **Hermes Agent** | `hermes model` → Custom endpoint → base URL `http://127.0.0.1:4000/v1`, model `auto` (needs a long-context model: pin one with `large` or a concrete id) |
| **Aider** | `aider --openai-api-base http://127.0.0.1:4000/v1 --openai-api-key freelm --model openai/auto` |
| **Open WebUI** | Admin → Connections → OpenAI API: `http://host.docker.internal:4000/v1` |
| **n8n** | OpenAI credential → Base URL `http://127.0.0.1:4000/v1` |

Or with Docker (image built on each release):

```bash
docker run --rm -p 4000:4000 -e GEMINI_API_KEY=... -e FREELM_SERVER_KEY=change-me ghcr.io/shihabshahrier/freelm
```

`GET /v1/models` lists the aliases plus every discovered free model; `GET /health` shows key state. A model id no
free provider offers (a tool's default `gpt-4o`) is tried, then served by `auto` (`--no-fallback` to disable). Bound
to localhost by default — if you expose it (`--host 0.0.0.0`, Docker on Linux), set `--api-key` so nobody else spends
your quota. In code: `freelm.serve()` (Python) / `serve()` (JS).

## Python

```python
from freelm import FreeLLM, GoogleAIStudio, Groq, OpenRouter

llm = FreeLLM(
    [GoogleAIStudio("AIza..."), Groq("gsk_..."), OpenRouter("sk-or-...")],   # or FreeLLM.from_env()
    strategy="smart",         # smart (default) | priority | round_robin | quota_aware | latency
)

r = llm.chat([{"role": "user", "content": "Write a haiku about failover."}], model="chat:fast")
print(r.text, "via", r.provider, r.model)

for chunk in llm.stream("Count to five."):          # token streaming, failover before the first token
    print(chunk, end="", flush=True)

r = llm.chat(msgs, model="chat:tools", tools=[...], tool_choice="auto")   # function calling
r.tool_calls

llm.chat(msgs, response_format={"type": "json_object"})                    # JSON mode
```

Async is symmetric (`AsyncFreeLLM`, `await llm.chat(...)`, `async for c in llm.astream(...)`). For raw OpenAI
`chat.completion.chunk` objects — tool-call deltas, finish reasons, usage — use `stream_chunks()` /
`astream_chunks()`.

**Drop-in for the OpenAI SDK** — change one import:

```python
# from openai import OpenAI
from freelm.compat import OpenAI

client = OpenAI()   # api_key / base_url / ... are accepted and ignored; keys come from the environment
r = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}])
print(r.choices[0].message.content)
```

Responses behave like the SDK's (`r.choices[0].message.tool_calls[0].function.arguments`, `model_dump()`,
`stream=True` chunks, `with` blocks, `client.models.list()`).

## JavaScript / TypeScript

```ts
import { FreeLLM } from "freelm";

const llm = FreeLLM.fromEnv();
console.log(await llm.text("Explain failover in one sentence."));
for await (const chunk of llm.stream("Count to five.", { signal: AbortSignal.timeout(30_000) })) process.stdout.write(chunk);
```

Same engine, same providers, zero dependencies; ESM + CommonJS. No `node:` imports in the library, so it also
bundles for edge functions, Workers and browsers (disk cache/persistence switch on only where Node built-ins exist).
See the
[JS README](https://github.com/shihabshahrier/freelm/tree/main/js#readme).

## Free providers

| Provider | Get a free key | Environment variable | Notes |
|----------|---------------|----------------------|-------|
| Google AI Studio (Gemini) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `GEMINI_API_KEY` | Most generous free tier; per-model limits. Pro models have no free quota. |
| Groq | [console.groq.com/keys](https://console.groq.com/keys) | `GROQ_API_KEY` | Very fast; per model 30 req/min, 1K req/day. Not xAI's "Grok" (that one is paid). |
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) | `OPENROUTER_API_KEY` | `:free` models + the `openrouter/free` router only (guarded); 20 req/min, 50 req/day (1000/day after $10 of lifetime credit). |
| Cloudflare Workers AI (beta) | [dash.cloudflare.com](https://dash.cloudflare.com/profile/api-tokens) | `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` | 10,000 Neurons/day free on every account (a few hundred chats); Llama 4 Scout, gpt-oss, Qwen 3.8, Gemma 4, GLM-4.7-Flash. Token needs Workers AI permission. |
| Z.ai (GLM) (beta) | [z.ai](https://z.ai/manage-apikey/apikey-list) | `ZAI_API_KEY` | GLM-4.7-Flash, GLM-4.5-Flash and GLM-4.6V-Flash (vision) are free; other GLM models are paid, so they're blocked (guarded). |
| Cohere (beta) | [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) | `COHERE_API_KEY` | Command A / A+ on a free **trial** key: 20 req/min per model, 1,000 calls/month, non-commercial use. |
| Kilo Gateway | [app.kilo.ai](https://app.kilo.ai) (optional) | `KILO_API_KEY` | Free routes (`:free`, `kilo-auto/free`) work **without a key** (~200 req/hour per IP); a free account lifts that. Free-only guard on. |
| OVHcloud AI Endpoints | none — anonymous | — | Keyless, 2 req/min per IP per model; last-resort fallback. (An OVH key is pay-as-you-go, so freelm doesn't use one.) |
| Cerebras | [cloud.cerebras.ai](https://cloud.cerebras.ai) | `CEREBRAS_API_KEY` | ⚠️ No longer permanently free (trial credits, card required) — supported if you already have a key. |
| Mistral | [console.mistral.ai](https://console.mistral.ai/api-keys) | `MISTRAL_API_KEY` | "Experiment" plan; low requests/minute. |
| NVIDIA NIM | [build.nvidia.com](https://build.nvidia.com/settings/api-keys) | `NVIDIA_API_KEY` | Free against build credits. |

*Beta* = added in 0.5.0 and not yet verified with live keys — `freelm doctor` tells you if yours works.
Several keys per provider: comma-separate them or use `FREELM_<PROVIDER>_KEYS`. The library never uses the
keyless endpoints unless asked — `FreeLLM.from_env(keyless=True)` (always, as last resorts), `keyless="auto"`
(only when no keys are set — the CLI's default) or `FREELM_KEYLESS=1|auto|0`. Any other OpenAI-compatible endpoint
works too: `Provider("key", name="myhost", base_url="https://.../v1", models=[ModelSpec("model-id", ("chat",))])`.

Free tiers change constantly — models get retired, limits move, keys expire. freelm is built for that: it discovers
models live (OpenRouter, Groq, Mistral, Kilo, OVHcloud, Cerebras), benches retired or overloaded models instead of retrying them,
and `freelm doctor` tells you what works right now. The built-in fallback model lists were re-verified on 2026-10-09.

## How it stays up

- **A slow or hung provider doesn't hold the call.** When an attempt is still running after a few seconds (3× that
  provider's usual latency, clamped; ~3 s for a stream's first token, ~6 s for a whole answer when it's unknown), the
  next provider starts in parallel and the first answer wins — the slow one is cancelled and remembered.
- **Smart routing (default):** providers are ranked by measured latency, so a fast one serves first and a slow one
  drops back; an unknown provider counts as typical so it gets tried, and old measurements expire after 10 minutes.
- **Every failure fails over.** 429 → rotate the key (or just bench the model, where quotas are per-model: Gemini,
  Groq, Z.ai, Cohere; a key whose daily or monthly quota is spent rests instead of being retried every minute);
  5xx / timeouts → circuit breaker + backoff; 401/402 → disable that key; a retired model (404/410) → bench it
  for an hour; one provider rejecting a request (unsupported parameter, moderation) → try the next provider. Only a
  request that several providers reject is reported as your bug.
- **Breadth-first failover:** the best model of *every* provider is tried before any provider's second model, so a
  provider full of throttled or dead models can't stall the call.
- **Quota guard:** per-key requests/minute bucket + daily counter; keys predicted to be exhausted are skipped.
- **Errors that explain themselves:** when everything fails, `NoProvidersAvailable` says why per provider
  (`openrouter: disabled (auth:401 — key invalid/expired?) | google: cooling down ~30s ...`).

## Choosing models

| Alias | Meaning |
|-------|---------|
| `auto` / `chat` | best available chat model (non-thinking models first) |
| `chat:fast` · `chat:large` · `chat:small` | by size/speed |
| `chat:tools` · `vision` | only models that support function calling / images |
| `reasoning` | prefer thinking models |
| `vendor/model-id` | exactly this model, routed only to providers that list it |
| `["vendor/id", "chat:fast"]` | an ordered per-call fallback chain |

Bias selection with `prefer=["gemini-2.5-flash", "gpt-oss"]` (exact id or substring), order a static list with
`ModelSpec(priority=)`, or rank providers with `priority=` (lower = first).

> **Thinking models** (gemini-2.5-flash, gpt-oss, ...) can spend a small `max_tokens` budget on hidden reasoning and
> return empty text — give them ≥ 128 tokens or use `auto`, which prefers non-thinking models.

## Configuration

| `FreeLLM(...)` option | Default | |
|---|---|---|
| `strategy` | `"smart"` | `smart` (priority tiers, then fastest measured) · `priority` · `round_robin` · `quota_aware` · `latency` |
| `hedge` | `True` | race a slow attempt against the next provider: `True` = adaptive delay, seconds = fixed, `False` = one at a time |
| `max_attempts` | `12` | cap on tries across providers/keys/models per call |
| `timeout` | `60` | seconds; also the overall deadline for one call |
| `wait` / `max_wait` | `False` / `20` | sleep until a key frees up instead of failing |
| `on_event` | — | callback for `attempt` / `hedge` / `success` / `error` / `wait` / `discovery` events (keys masked) |
| `persist` | `False` (`FREELM_PERSIST=1`) | keep quota/cooldown/disabled-key state across restarts (`~/.cache/freelm/state.json`, 0600, key hashes only) |

Provider options: `tier`, `priority`, `prefer`, `models`, `rpm`/`rpd`, `discover`, `free_only` (OpenRouter defaults
to `True`: paid ids raise `ConfigError` instead of billing you), `cache_ttl`. `llm.health()` shows per-key state;
`llm.reset_keys()` gives every key a fresh start.

## CLI

```bash
freelm chat "explain failover in one line" --model chat:fast --stream
freelm doctor [--json]          # live-check every key, with fixes
freelm serve [--port 4000] [--api-key KEY] [--cors]
freelm models --provider groq   # live model list
freelm health                   # this process's key state
```

## When can freelm cost money?

Free-only by default and by guard: OpenRouter, Kilo and Z.ai paid models are blocked unless you pass
`free_only=False`; Google is free unless *you* pick `tier="tier1"` (billing enabled); NIM consumes free build credits
and Cerebras trial credits (requests fail when they run out — nothing is billed through freelm); Groq and Mistral
free accounts only have free models; OVHcloud is used anonymously only. Your account's plan decides the rest: Cohere
bills only production (non-trial) keys, and Cloudflare only on the Workers Paid plan (past the daily free Neurons —
the Free plan just stops).

## How freelm compares

| Project | What it is | Difference |
|---------|------------|------------|
| [freellmapi](https://github.com/tashfeenahmed/freellmapi), [freellmpool](https://github.com/0xzr/freellmpool) | Self-hosted gateways pooling many free providers | freelm is a library first (`pip`/`npm install`, nothing to host) with the gateway optional (`freelm serve`), in both Python and TypeScript |
| [LiteLLM](https://github.com/BerriAI/litellm) | SDK + proxy for 100+ providers, paid and free | freelm is free-only, zero-dependency, with per-key quota/breaker state and free-model discovery built in |
| [OpenRouter](https://openrouter.ai) | One aggregator | One of freelm's pools — when its free quota runs out, freelm fails over to Gemini, Groq, Cloudflare, Z.ai, Mistral or NIM directly |
| LangChain / LlamaIndex | Orchestration frameworks | Use freelm under them via `freelm serve` or the OpenAI-compatible shim |

## FAQ

### What is the best free LLM API in 2026?
Google AI Studio (Gemini) has the most generous free tier; Groq is the fastest; OpenRouter has the widest choice of
`:free` models. Each has its own limits and outages, so the most reliable option is to use several at once —
freelm pools them behind one OpenAI-compatible call and fails over automatically.

### Is there a free alternative to the OpenAI API?
Yes. `freelm.compat.OpenAI` is a drop-in for the OpenAI SDK, and `freelm serve` exposes a local
`/v1/chat/completions` endpoint, both backed by free tiers (Gemini, Groq, OpenRouter, Cloudflare Workers AI, Z.ai,
Cohere, Mistral, NVIDIA NIM)
with streaming, tool calling and automatic failover.

### How do I use free LLMs in OpenCode, OpenClaw, Hermes, Cline, Continue, Open WebUI or n8n?
Run `freelm serve` and set the tool's OpenAI base URL to `http://127.0.0.1:4000/v1` with model `auto` — see the
[table above](#use-free-llms-in-any-openai-compatible-tool). Tools that call the endpoint from their own servers need
a public URL (for example a tunnel) plus `--api-key`.

### How do I use free LLMs in Python or JavaScript?
`pip install freelm` or `npm install freelm`, set one free key (e.g. `GEMINI_API_KEY`), then
`freelm.FreeLLM.from_env().text("...")` / `await FreeLLM.fromEnv().text("...")`.

### Can I use an LLM API with no API key at all?
Yes, for trying things out: `freelm chat "hello"` (or `npx freelm chat "hello"`) works with no keys by using
documented keyless endpoints — Kilo Gateway's free routes (~200 requests/hour per IP) and OVHcloud AI Endpoints'
anonymous tier. Limits are low and free routes may log prompts; one free Gemini or Groq key is far more capable.

### Why did my free model stop working?
Free model ids are retired constantly (NVIDIA retired its Llama 3.x endpoints on 2026-08-26; Groq retired Llama 3.3 70B
on 2026-08-16). freelm discovers current models live, benches retired ones on the first 404/410 and fails over, so
your code keeps working; `freelm doctor` shows which keys and models work today.

### How fast is failover?
An error (429, 5xx, bad key, retired model) fails over immediately — the cost is one extra round trip, usually well
under a second. A provider that hangs or answers slowly is raced: after a few seconds the next provider starts in
parallel and the first answer wins (measured: a hung provider ahead of a healthy one answers in ~6 s, a stalled
stream in ~3 s). After that, smart routing sends calls to the fast provider first, so later calls don't wait at all.
The numbers come from a reproducible simulation: [`benchmarks/`](https://github.com/shihabshahrier/freelm/tree/main/benchmarks).
A host that won't accept a connection fails within 10 s.

### How do I avoid free-tier rate limits (429)?
Add more providers and keys: freelm paces each key, rotates on 429, benches per-model quotas (Gemini, Groq) and
spreads load with `strategy="quota_aware"`.

### Is freelm really free? Does it see my prompts?
freelm is MIT-licensed and runs in your process (or on your machine with `freelm serve`); requests go straight from
you to the providers whose keys you configure. It has no telemetry. Usage limits are those of each provider's free
tier.

## Links

[Changelog](https://github.com/shihabshahrier/freelm/blob/main/CHANGELOG.md) ·
[Contributing](https://github.com/shihabshahrier/freelm/blob/main/CONTRIBUTING.md) ·
[Security](https://github.com/shihabshahrier/freelm/blob/main/SECURITY.md) ·
[Website & docs](https://shihub.site/freelm) · [PyPI](https://pypi.org/project/freelm/) ·
[npm](https://www.npmjs.com/package/freelm)

MIT © Shahriar Labs
