import { ErrorScope } from "../errors.js";
import { modelSpec, ModelSpec } from "../registry.js";
import { Provider, TierLimit } from "./base.js";

/** Cohere via its OpenAI compatibility API.
 *
 * Free **trial** keys (no card): 20 chat requests/minute per model and 1,000
 * calls/month, for non-commercial use (checked 2026-10). Production keys are
 * billed — use a trial key with freelm. A per-minute 429 benches just that
 * model; the monthly cap's 429 disables the key (`classify`). Cohere rejects a
 * few OpenAI parameters (`n`, `logit_bias`, `parallel_tool_calls`, ...); freelm
 * fails such requests over to another provider. */
export class Cohere extends Provider {
  static providerName = "cohere";
  static baseUrl = "https://api.cohere.ai/compatibility/v1";
  // trial key: 20 RPM per model; the 1,000 calls/month cap can't be paced per day
  static tiers: Record<string, TierLimit> = { free: { rpm: 20, rpd: null } };
  // "Live" chat models on 2026-10-09 (docs.cohere.com/docs/models).
  static defaultModels: ModelSpec[] = [
    modelSpec("command-a-plus-05-2026", ["chat", "large", "tools", "vision"], 128000),
    modelSpec("command-a-03-2025", ["chat", "large", "tools"], 256000),
    modelSpec("command-a-reasoning-08-2025", ["chat", "large", "tools", "reasoning"], 256000),
    modelSpec("command-a-vision-07-2025", ["chat", "vision"], 128000),
    modelSpec("command-r7b-12-2024", ["chat", "small", "fast", "tools"], 128000),
  ];

  rateLimitScope(_body: string): ErrorScope {
    return "model"; // trial limits are per model
  }
}
