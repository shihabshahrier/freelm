# freelm — free LLM API for Node.js & TypeScript (OpenAI-compatible, auto-failover)

[![npm](https://img.shields.io/npm/v/freelm?label=npm)](https://www.npmjs.com/package/freelm)
[![CI](https://github.com/shihabshahrier/freelm/actions/workflows/js-ci.yml/badge.svg)](https://github.com/shihabshahrier/freelm/actions/workflows/js-ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/shihabshahrier/freelm/blob/main/LICENSE)

**freelm turns the free tiers of Google Gemini, Groq, OpenRouter, Cloudflare Workers AI, Z.ai, Cohere, Mistral and
NVIDIA NIM into one OpenAI-compatible LLM — in your TypeScript code, or as a local `/v1` endpoint for any tool (`npx freelm serve`).**
A free-only, lightweight alternative to LiteLLM or an OpenRouter account: it rotates your keys, falls back across
providers on rate limits, outages and retired models, races a slow provider against the next one (a hung provider
costs ~6 s, not a 60 s failure), and discovers which models are free today. Zero dependencies, ESM + CommonJS, your
own keys called directly.

> Also on PyPI with the same engine: [`pip install freelm`](https://pypi.org/project/freelm/).

```bash
npm install freelm        # Node >= 20
```

```ts
import { FreeLLM } from "freelm";

const llm = FreeLLM.fromEnv();     // reads GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, ...
console.log(await llm.text("Explain failover in one sentence."));
```

## Start in 60 seconds

**Zero keys?** `npx freelm chat "hello"` already works: with no keys set, the CLI falls back to keyless public
endpoints (Kilo Gateway free routes, OVHcloud anonymous) and says so — low limits, free routes may log prompts. In
code that's opt-in: `FreeLLM.fromEnv({ keyless: "auto" })` or `FREELM_KEYLESS=1`.

1. Get one free key (no card): [Google AI Studio](https://aistudio.google.com/apikey); add
   [Groq](https://console.groq.com/keys) / [OpenRouter](https://openrouter.ai/keys) for failover.
2. `export GEMINI_API_KEY=...`
3. `npx freelm doctor` — one tiny request per key; says exactly what's wrong and where to get a new key.

## Local OpenAI-compatible endpoint

```bash
npx freelm serve          # → http://127.0.0.1:4000/v1  (/v1/chat/completions with SSE, /v1/models, /health)
```

Point the OpenAI SDK, LangChain, LlamaIndex, Continue, Cline, Aider, Open WebUI or n8n at
`http://127.0.0.1:4000/v1` with any API key and model `auto`. With the Vercel AI SDK:

```ts
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { generateText } from "ai";

const freelm = createOpenAICompatible({ name: "freelm", baseURL: "http://127.0.0.1:4000/v1" });
const { text } = await generateText({ model: freelm("auto"), prompt: "hi" });
```

Embed it instead of using the CLI: `import { serve } from "freelm"; await serve({ port: 4000 });`. It binds to
localhost; set `apiKey` / `--api-key` before exposing it.

## Usage

```ts
import { FreeLLM, GoogleAIStudio, Groq, OpenRouter } from "freelm";

const llm = new FreeLLM(
  [new GoogleAIStudio("AIza..."), new Groq("gsk_..."), new OpenRouter("sk-or-...")],
  { strategy: "smart" },   // smart (default) | priority | round_robin | quota_aware | latency
);

const r = await llm.chat([{ role: "user", content: "Write a haiku about failover." }], { model: "chat:fast" });
console.log(r.text, "via", r.provider, r.model);

// streaming — fails over before the first token; cancel with an AbortSignal
for await (const chunk of llm.stream("Count to five.", { signal: AbortSignal.timeout(30_000) })) process.stdout.write(chunk);

// raw chat.completion.chunk objects (tool-call deltas, finish_reason, usage)
for await (const c of llm.streamChunks(msgs, { tools })) console.log(c.choices[0].delta);

// tools / JSON mode pass straight through
const t = await llm.chat(msgs, { model: "chat:tools", tools, tool_choice: "auto" });
t.toolCalls;
```

**Drop-in for the OpenAI SDK:**

```ts
// import OpenAI from "openai";
import { OpenAI } from "freelm/compat";

const client = new OpenAI();   // { apiKey, baseURL, ... } accepted and ignored; keys come from the environment
const r = await client.chat.completions.create({ model: "auto", messages: [{ role: "user", content: "hi" }] });
const stream = await client.chat.completions.create({ model: "auto", messages, stream: true }, { signal });
for await (const chunk of stream) process.stdout.write(chunk.choices[0].delta.content ?? "");
```

## Providers & environment

| Provider | Free key | Variable (comma-separate for several keys) |
|----------|----------|------------------------------------------|
| Google AI Studio | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `FREELM_GOOGLE_KEYS` |
| Groq | [console.groq.com/keys](https://console.groq.com/keys) | `GROQ_API_KEY` / `FREELM_GROQ_KEYS` |
| OpenRouter (`:free` models) | [openrouter.ai/keys](https://openrouter.ai/keys) | `OPENROUTER_API_KEY` / `FREELM_OPENROUTER_KEYS` |
| Cloudflare Workers AI (beta; 10,000 Neurons/day) | [dash.cloudflare.com](https://dash.cloudflare.com/profile/api-tokens) | `CLOUDFLARE_API_TOKEN` / `FREELM_CLOUDFLARE_KEYS` **+** `CLOUDFLARE_ACCOUNT_ID` |
| Z.ai (beta; free GLM Flash models only, guarded) | [z.ai](https://z.ai/manage-apikey/apikey-list) | `ZAI_API_KEY` / `FREELM_ZAI_KEYS` |
| Cohere (beta; trial key: free, non-commercial) | [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) | `COHERE_API_KEY` / `CO_API_KEY` / `FREELM_COHERE_KEYS` |
| Kilo Gateway (works keyless) | [app.kilo.ai](https://app.kilo.ai) | `KILO_API_KEY` / `FREELM_KILO_KEYS` (optional) |
| OVHcloud AI Endpoints | none — anonymous only | — |
| Cerebras (trial credits, no longer permanently free) | [cloud.cerebras.ai](https://cloud.cerebras.ai) | `CEREBRAS_API_KEY` / `FREELM_CEREBRAS_KEYS` |
| Mistral | [console.mistral.ai](https://console.mistral.ai/api-keys) | `MISTRAL_API_KEY` / `FREELM_MISTRAL_KEYS` |
| NVIDIA NIM | [build.nvidia.com](https://build.nvidia.com/settings/api-keys) | `NVIDIA_API_KEY` / `NIM_API_KEY` / `FREELM_NIM_KEYS` |

Any other OpenAI-compatible endpoint: `new Provider("key", { name: "myhost", baseUrl: "https://.../v1", models: [modelSpec("id", ["chat"])] })`.

## Models

`"auto"` / `"chat"` (best available, non-thinking first), `"chat:fast"`, `"chat:large"`, `"chat:small"`,
`"chat:tools"` / `"vision"` (only capable models), `"reasoning"`, a concrete id (routed only to providers that list
it), or an ordered fallback list `["vendor/id", "chat:fast"]`. Bias discovered lists with `prefer: [...]`; rank
providers with `priority` (lower = first). Free model ids churn, so freelm discovers them live and benches retired or
overloaded models on the first 404/410/503 instead of retrying them.

## Reliability, observability, persistence

- A slow or hung provider doesn't hold the call: after a few seconds (3× its usual latency, clamped; ~3 s for a
  stream's first token) the next provider starts in parallel and the first answer wins — the slow request is aborted
  and remembered. `{ hedge: false }` = one attempt at a time; `{ hedge: 2 }` = fixed 2 s.
- Smart routing (default `strategy`): providers ranked by measured latency; unknown ones count as typical so they get
  tried; measurements expire after 10 minutes.
- Every failure fails over: 429 → rotate key (or bench just the model where quotas are per-model); 5xx/timeouts →
  breaker + backoff; 401/402 → disable key; retired model → bench it; one provider rejecting a request → next
  provider (raised only if two providers reject it). Breadth-first across providers, so none can stall a call.
- `new FreeLLM(provs, { onEvent: (e) => console.log(e.kind, e.provider, e.model, e.status) })` — keys always masked.
- `{ persist: true }` (or `FREELM_PERSIST=1`) keeps quota/cooldown/disabled-key state across restarts
  (`~/.cache/freelm/state.json`, 0600, key hashes only). `llm.health()`, `llm.resetKeys()`.
- No `node:` imports in the library: it bundles for edge functions, Workers and browsers (disk cache/persistence
  switch on only where Node built-ins exist).

## CLI

```bash
npx freelm chat "explain failover in one line" --model chat:fast --stream
npx freelm doctor [--json]
npx freelm serve [--port 4000] [--api-key KEY] [--cors] [--no-fallback]
npx freelm models --provider groq
npx freelm health
```

## FAQ

**What is the best free LLM API for Node.js?** Google AI Studio (Gemini) has the most generous free tier, Groq is
the fastest, OpenRouter has the most `:free` models — and each has outages and limits. freelm uses them together
behind one OpenAI-compatible call, so you get whichever is up and fast right now.

**How fast is failover?** An error (429, 5xx, bad key, retired model) moves to the next provider immediately — one
extra round trip. A provider that hangs or answers slowly is raced: after a few seconds the next one starts in
parallel and the first answer wins (a hung provider costs ~6 s once, a stalled stream ~3 s); smart routing then
sends later calls to the fast provider first.

**Can I use it without an API key?** Yes, for trying it out: `npx freelm chat "hello"` falls back to keyless public
endpoints (Kilo Gateway, OVHcloud) and says so. One free Gemini or Groq key is far more capable.

**When can freelm cost money?** It is free-only by default: OpenRouter, Kilo and Z.ai paid models are blocked unless
you pass `freeOnly: false`. Your own account's plan decides the rest (Gemini with billing enabled, Cohere production
keys, Cloudflare's Workers Paid plan past the daily free Neurons).

**freelm vs LiteLLM or OpenRouter?** LiteLLM covers 100+ paid and free providers; freelm is free-only, zero-dependency
and focused on surviving free-tier limits (per-key quotas, model benching, live free-model discovery). OpenRouter is
one aggregator — freelm uses it as one pool and fails over to Gemini, Groq, Cloudflare and others directly.

Full docs and a comparison with other free-LLM gateways: the
[main README](https://github.com/shihabshahrier/freelm#readme) ·
[Changelog](https://github.com/shihabshahrier/freelm/blob/main/CHANGELOG.md).

MIT © Shahriar Labs
