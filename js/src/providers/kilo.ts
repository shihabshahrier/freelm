import { ANONYMOUS } from "../keys.js";
import { modelSpec, ModelSpec } from "../registry.js";
import { Provider, ProviderOptions, TierLimit } from "./base.js";

/** Kilo Gateway — OpenAI-compatible gateway whose free routes work **without
 * any key** (about 200 requests/hour per IP, verified 2026-10); a free Kilo
 * account key lifts that limit. Its catalog mixes paid and free models, so like
 * OpenRouter it is free-only by default. Free routes may log prompts — the
 * catalog's `mayTrainOnYourPrompts` flag says which. */
export class Kilo extends Provider {
  static providerName = "kilo";
  static baseUrl = "https://api.kilo.ai/api/gateway";
  static keyless = true;
  // anonymous: ~200 requests/hour per IP -> pace to 3/min
  static tiers: Record<string, TierLimit> = { free: { rpm: 3, rpd: null } };
  // Free routes answering keyless on 2026-10-09; discovery replaces the list.
  static defaultModels: ModelSpec[] = [
    modelSpec("poolside/laguna-s-2.1:free", ["chat", "tools"], 262144),
    modelSpec("nvidia/nemotron-3-super-120b-a12b:free", ["chat", "large", "tools", "reasoning"], 262144),
    modelSpec("kilo-auto/free", ["chat", "tools", "reasoning"], 256000),
    modelSpec("openrouter/free", ["chat"], 200000),
  ];

  constructor(keys?: string | string[] | null, opts: ProviderOptions = {}) {
    super(keys && keys.length ? keys : ANONYMOUS, { discover: !opts.models, discoverFreeOnly: true, freeOnly: true, ...opts });
  }
}
