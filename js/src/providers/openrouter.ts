import { ErrorScope } from "../errors.js";
import { modelSpec, ModelSpec } from "../registry.js";
import { Provider, ProviderOptions, TierLimit } from "./base.js";

/** OpenRouter — OpenAI-compatible. Free models carry a `:free` suffix.
 * Live `/models` discovery on by default (free model ids churn constantly). */
export class OpenRouter extends Provider {
  static providerName = "openrouter";
  static baseUrl = "https://openrouter.ai/api/v1";
  static tiers: Record<string, TierLimit> = {
    free: { rpm: 20, rpd: 50 }, // < $10 lifetime credit
    credit: { rpm: 20, rpd: 1000 }, // >= $10 lifetime credit
  };
  // Free model ids churn constantly; these were listed as free in the public
  // catalog on 2026-10-09 (diverse upstreams, so one throttle still fails over).
  // Live discovery replaces this list at runtime.
  static defaultModels: ModelSpec[] = [
    modelSpec("google/gemma-4-31b-it:free", ["chat", "large", "tools", "vision"], 262144),
    modelSpec("nvidia/nemotron-3-super-120b-a12b:free", ["chat", "large", "tools", "reasoning"], 262144),
    modelSpec("google/gemma-4-26b-a4b-it:free", ["chat", "fast", "tools", "vision"], 262144),
    modelSpec("nvidia/nemotron-3.5-lightning:free", ["chat", "fast", "tools"], 1000000),
    modelSpec("poolside/laguna-s-2.1:free", ["chat", "tools"], 262144),
    modelSpec("thinkingmachines/inkling-small:free", ["chat", "small", "fast", "tools", "vision"], 1048576),
    // OpenRouter's own router across whatever free models are up right now
    modelSpec("openrouter/free", ["chat"], 200000),
  ];

  constructor(keys: string | string[], opts: ProviderOptions = {}) {
    // App attribution: OpenRouter lists apps that send a referer + title on
    // openrouter.ai/apps and in each model's "Apps" tab (X-Title is the legacy
    // name of X-OpenRouter-Title; categories: at most 2 per request, from
    // openrouter.ai/docs/app-attribution). Override via extraHeaders.
    const extraHeaders = {
      "HTTP-Referer": "https://github.com/shihabshahrier/freelm",
      "X-OpenRouter-Title": "freelm",
      "X-Title": "freelm",
      "X-OpenRouter-Categories": "programming-app,general-chat",
      ...(opts.extraHeaders ?? {}),
    };
    // OpenRouter's catalog mixes paid and free models -> guard paid ids by default.
    super(keys, { discover: !opts.models, discoverFreeOnly: true, freeOnly: true, ...opts, extraHeaders });
  }

  rateLimitScope(body: string): ErrorScope {
    // e.g. "<model> is temporarily rate-limited upstream". Deliberately narrow:
    // a bare "temporarily" also appears in account-wide 429s, which must cool
    // the key instead of hammering it with the next model.
    const b = (body || "").toLowerCase();
    return b.includes("rate-limited upstream") ? "upstream" : "key"; // throttled for everyone, not just this key
  }
}
