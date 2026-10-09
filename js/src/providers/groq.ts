import { ErrorScope } from "../errors.js";
import { modelSpec, ModelSpec } from "../registry.js";
import { Provider, ProviderOptions, TierLimit } from "./base.js";

/** Groq — OpenAI-compatible, very fast inference, free dev tier.
 * Free tier (console.groq.com/docs/rate-limits, checked 2026-10): per model
 * 30 RPM, 1K RPD, 8K TPM, 200K TPD. No credit card required. */
export class Groq extends Provider {
  static providerName = "groq";
  static baseUrl = "https://api.groq.com/openai/v1";
  static tiers: Record<string, TierLimit> = {
    free: { rpm: 30, rpd: 1000 },
  };
  // llama-3.3-70b-versatile and llama-3.1-8b-instant were shut down for free/dev
  // accounts on 2026-08-16; these are Groq's named successors (deprecations
  // page, checked 2026-10). Live discovery replaces this list at runtime.
  static defaultModels: ModelSpec[] = [
    modelSpec("qwen/qwen3.6-27b", ["chat", "tools"]),
    modelSpec("openai/gpt-oss-120b", ["chat", "large", "tools", "reasoning"], 131072),
    modelSpec("openai/gpt-oss-20b", ["chat", "small", "fast", "tools", "reasoning"], 131072),
  ];

  constructor(keys: string | string[], opts: ProviderOptions = {}) {
    // live /models self-corrects the list — unless the caller pinned models
    super(keys, { discover: !opts.models, ...opts });
  }

  /** Groq limits (RPM/RPD/TPM/TPD) apply per model, so a 429 on one model
   * leaves the others usable on the same key. */
  rateLimitScope(_body: string): ErrorScope {
    return "model";
  }
}
