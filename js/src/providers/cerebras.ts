import { modelSpec, ModelSpec } from "../registry.js";
import { Provider, ProviderOptions, TierLimit } from "./base.js";

/** Cerebras — OpenAI-compatible, very fast inference.
 * As of 2026-10 Cerebras has no permanently free tier: new accounts get trial
 * credits (payment method required). freelm still supports a key you already
 * have; when the credits run out requests fail with 402 and the key is
 * disabled — nothing is billed through freelm. */
export class Cerebras extends Provider {
  static providerName = "cerebras";
  static baseUrl = "https://api.cerebras.ai/v1";
  static tiers: Record<string, TierLimit> = {
    free: { rpm: 30, rpd: null },
  };
  // The live catalog (2026-10-09) lists only these two; llama-3.3-70b and
  // qwen-3-32b are gone (404). Runtime discovery replaces this list.
  static defaultModels: ModelSpec[] = [
    modelSpec("qwen-3.8-27b", ["chat", "small", "fast"], 8192),
    modelSpec("gpt-oss-120b", ["chat", "large", "reasoning"], 8192),
  ];

  constructor(keys: string | string[], opts: ProviderOptions = {}) {
    super(keys, { discover: !opts.models, ...opts });
  }
}
