import { modelSpec, ModelSpec } from "../registry.js";
import { Provider, ProviderOptions, TierLimit } from "./base.js";

/** Cerebras — OpenAI-compatible, fast inference, free tier.
 * Free tier (verified 2026-06): ~30 RPM, 1,000,000 tokens/day, 8,192 ctx cap.
 * Token-limited, not request/day-limited, so rpd is null. */
export class Cerebras extends Provider {
  static providerName = "cerebras";
  static baseUrl = "https://api.cerebras.ai/v1";
  static tiers: Record<string, TierLimit> = {
    free: { rpm: 30, rpd: null },
  };
  static defaultModels: ModelSpec[] = [
    modelSpec("llama-3.3-70b", ["chat", "large"], 8192),
    modelSpec("qwen-3-32b", ["chat", "large"], 8192),
    modelSpec("gpt-oss-120b", ["chat", "large", "reasoning"], 8192),
  ];

  constructor(keys: string | string[], opts: ProviderOptions = {}) {
    super(keys, { discover: true, ...opts });
  }
}
