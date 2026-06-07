# freelm (JS/TS)

[![npm](https://img.shields.io/npm/v/freelm.svg)](https://www.npmjs.com/package/freelm)
[![license](https://img.shields.io/npm/l/freelm.svg)](../LICENSE)

**freelm is a free, always-up LLM client for Node.js/TypeScript** that pools multiple free-tier LLM providers — **OpenRouter, Google Gemini (AI Studio), NVIDIA NIM, Groq, Cerebras, and Mistral** — behind one OpenAI-compatible call (with streaming), with automatic key rotation, cross-provider failover, circuit breaking, and live free-model discovery. Drop in whichever free keys you have and your app keeps talking to an LLM even when one source rate-limits or goes down.

> The TypeScript port of [freelm for Python](https://pypi.org/project/freelm/) — same API, same behavior. Zero runtime dependencies (uses the built-in `fetch`).

## Install

```bash
npm install freelm
```

## Quick start

```ts
import { FreeLLM } from "freelm";

const llm = FreeLLM.fromEnv();                 // reads provider keys from env
console.log(await llm.text("Explain black holes in one sentence."));
```

Explicit config:

```ts
import { FreeLLM, OpenRouter, GoogleAIStudio, NIM, Groq, Cerebras, Mistral } from "freelm";

const llm = new FreeLLM(
  [
    new OpenRouter("sk-or-..."),
    new GoogleAIStudio("AIza..."),
    new Groq("gsk_..."),
    new Cerebras("csk-..."),
    new Mistral("..."),
    new NIM("nvapi-..."),
  ],
  { strategy: "quota_aware" },                  // priority | round_robin | quota_aware | latency
);

const r = await llm.chat([{ role: "user", content: "Write a haiku about failover." }], { model: "chat:fast" });
console.log(r.text, "via", r.provider);
```

## Streaming

```ts
for await (const chunk of llm.stream("Stream me some tokens")) {
  process.stdout.write(chunk);
}
```

Streaming fails over between providers **before the first token**; once tokens flow it stays on that provider.

## Drop-in OpenAI shim

```ts
// import OpenAI from "openai";
import { OpenAI } from "freelm/compat";

const client = new OpenAI();                    // backed by FreeLLM.fromEnv()
const r = await client.chat.completions.create({
  model: "auto",
  messages: [{ role: "user", content: "hi" }],
});
console.log(r.choices[0].message.content);
```

## Environment variables

| Provider | Key vars (first match wins) | Tier var |
|----------|------------------------------|----------|
| OpenRouter | `OPENROUTER_API_KEY` / `FREELM_OPENROUTER_KEYS` | `FREELM_OPENROUTER_TIER` |
| Google AI Studio | `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `FREELM_GOOGLE_KEYS` | `FREELM_GOOGLE_TIER` |
| NVIDIA NIM | `NVIDIA_API_KEY` / `NIM_API_KEY` / `FREELM_NIM_KEYS` | `FREELM_NIM_TIER` |
| Groq | `GROQ_API_KEY` / `FREELM_GROQ_KEYS` | `FREELM_GROQ_TIER` |
| Cerebras | `CEREBRAS_API_KEY` / `FREELM_CEREBRAS_KEYS` | `FREELM_CEREBRAS_TIER` |
| Mistral | `MISTRAL_API_KEY` / `FREELM_MISTRAL_KEYS` | `FREELM_MISTRAL_TIER` |

Comma-separate to supply multiple keys per provider.

## Virtual models & discovery

Ask by intent — `"auto"`, `"chat:fast"`, `"chat:large"` — and freelm resolves each to a concrete model per provider. Free model ids churn, so freelm discovers them live from each provider's `/models` endpoint and caches them. List current free models:

```ts
import { listFreeModels } from "freelm";
for (const m of (await listFreeModels()).slice(0, 5)) console.log(m.id, m.tags);
```

## How "always-up" works

- **Key pool** per provider, rotated to spread load.
- **Failover** interleaved across providers, so every provider is reached fast.
- **Circuit breaker** per key — opens after repeated failures, half-opens after a cooldown.
- **Retry classification**: `429` → cool the key & rotate; `5xx`/timeout → backoff; `401` → disable the key; model errors → next model.
- **Quota guard**: per-key requests/minute + requests/day, skipping keys predicted exhausted.

Inspect live state with `llm.health()`.

## License

MIT © Shihab Shahriar Antor / [Shahriar Labs](https://shahriarlabs.com). Built by [Shihab Shahriar Antor](https://shihub.online). Python version: [pypi.org/project/freelm](https://pypi.org/project/freelm/).
