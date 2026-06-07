import { modelSpec, ModelSpec } from "../registry.js";
import { Provider, ProviderOptions, TierLimit } from "./base.js";

/** Mistral AI — OpenAI-compatible, free "Experiment" tier.
 * Free tier (verified 2026-06): 2 req/min, 500K tokens/min, 1B tokens/month.
 * Model ids self-correct via live /models discovery. */
export class Mistral extends Provider {
  static providerName = "mistral";
  static baseUrl = "https://api.mistral.ai/v1";
  static tiers: Record<string, TierLimit> = {
    free: { rpm: 2, rpd: null },
  };
  static defaultModels: ModelSpec[] = [
    modelSpec("mistral-small-latest", ["chat", "large", "tools"], 32000),
    modelSpec("mistral-large-latest", ["chat", "large", "tools"], 128000),
  ];

  constructor(keys: string | string[], opts: ProviderOptions = {}) {
    super(keys, { discover: true, ...opts });
  }
}
