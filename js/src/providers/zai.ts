import { ErrorScope } from "../errors.js";
import { modelSpec, ModelSpec } from "../registry.js";
import { Provider, ProviderOptions, TierLimit } from "./base.js";

/** Z.ai — GLM models over an OpenAI-compatible API.
 *
 * Only the *Flash* models are free (GLM-4.7-Flash, GLM-4.5-Flash, GLM-4.6V-Flash
 * per the official pricing page, checked 2026-10); every other GLM model is
 * billed against the account balance. So the provider is free-only with an
 * explicit list: a paid id throws ConfigError instead of spending your credit,
 * and discovery stays off (`/models` leaves the free Flash models out).
 *
 * The free models are limited by concurrent requests per model (GLM-4.7-Flash:
 * one at a time), so a 429 benches just that model for that key. GLM-4.7-Flash
 * thinks by default (`reasoning_content`) — give it room in `max_tokens`. */
export class ZAI extends Provider {
  static providerName = "zai";
  static baseUrl = "https://api.z.ai/api/paas/v4";
  // No published RPM: the free models are concurrency-limited (checked 2026-10).
  static tiers: Record<string, TierLimit> = { free: { rpm: 10, rpd: null } };
  static defaultModels: ModelSpec[] = [
    modelSpec("glm-4.7-flash", ["chat", "tools", "reasoning"], 200000),
    modelSpec("glm-4.5-flash", ["chat", "fast", "tools"], 128000),
    modelSpec("glm-4.6v-flash", ["chat", "vision", "tools"], 128000),
  ];

  constructor(keys: string | string[], opts: ProviderOptions = {}) {
    super(keys, { discover: false, freeOnly: true, ...opts });
  }

  /** 1302 (this account's concurrency for the model) and 1305 (the model is
   * overloaded) are both about one model; the other models stay usable. */
  rateLimitScope(_body: string): ErrorScope {
    return "model";
  }
}
