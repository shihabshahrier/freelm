import { modelSpec, ModelSpec } from "../registry.js";
import { Provider, ProviderOptions, TierLimit } from "./base.js";

/** Groq — OpenAI-compatible, very fast inference, free dev tier.
 * Free tier (verified 2026-06): 30 RPM, 14,400 req/day, resets midnight UTC. */
export class Groq extends Provider {
  static providerName = "groq";
  static baseUrl = "https://api.groq.com/openai/v1";
  static tiers: Record<string, TierLimit> = {
    free: { rpm: 30, rpd: 14400 },
  };
  static defaultModels: ModelSpec[] = [
    modelSpec("llama-3.3-70b-versatile", ["chat", "large", "tools"], 128000),
    modelSpec("llama-3.1-8b-instant", ["chat", "small", "fast"], 128000),
    modelSpec("openai/gpt-oss-120b", ["chat", "large", "tools", "reasoning"], 131072),
    modelSpec("openai/gpt-oss-20b", ["chat", "small", "fast", "tools", "reasoning"], 131072),
  ];

  constructor(keys: string | string[], opts: ProviderOptions = {}) {
    super(keys, { discover: true, ...opts });
  }
}
