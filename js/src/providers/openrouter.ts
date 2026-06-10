import { ConfigError } from "../errors.js";
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
  static defaultModels: ModelSpec[] = [
    modelSpec("openai/gpt-oss-120b:free", ["chat", "large"], 131072),
    modelSpec("openai/gpt-oss-20b:free", ["chat", "small", "fast"], 131072),
    modelSpec("meta-llama/llama-3.3-70b-instruct:free", ["chat", "large"], 131072),
    modelSpec("z-ai/glm-4.5-air:free", ["chat", "large"], 131072),
    modelSpec("qwen/qwen3-next-80b-a3b-instruct:free", ["chat", "large"], 262144),
    modelSpec("meta-llama/llama-3.2-3b-instruct:free", ["chat", "small", "fast"], 131072),
  ];

  constructor(keys: string | string[], opts: ProviderOptions = {}) {
    const extraHeaders = { "X-Title": "freelm", ...(opts.extraHeaders ?? {}) };
    // OpenRouter's catalog mixes paid and free models -> guard paid ids by default.
    super(keys, { discover: true, discoverFreeOnly: true, freeOnly: true, ...opts, extraHeaders });
  }

  protected checkFree(modelId: string): void {
    if (!this.freeOnly || modelId.endsWith(":free")) return;
    const spec = this.models.find((m) => m.id === modelId);
    if (spec && spec.free) return;
    throw new ConfigError(
      `[openrouter] '${modelId}' is not a ':free' model. freelm is free-only by default — ` +
        "pass new OpenRouter(key, { freeOnly: false }) to allow paid ids on your own account.",
    );
  }

  rateLimitScope(body: string): "key" | "model" {
    // e.g. "<model> is temporarily rate-limited upstream". Deliberately narrow:
    // a bare "temporarily" also appears in account-wide 429s, which must cool
    // the key instead of hammering it with the next model.
    const b = (body || "").toLowerCase();
    return b.includes("rate-limited upstream") ? "model" : "key";
  }
}
