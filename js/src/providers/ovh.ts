import { ErrorScope } from "../errors.js";
import { ANONYMOUS } from "../keys.js";
import { modelSpec, ModelSpec } from "../registry.js";
import { Provider, ProviderOptions, TierLimit } from "./base.js";

/** OVHcloud AI Endpoints — OpenAI-compatible and usable **anonymously**: 2
 * requests/minute per IP *per model* (verified 2026-10). An OVHcloud key
 * switches to pay-as-you-go, so freelm only uses the anonymous tier — a
 * last-resort, no-signup fallback. */
export class OVHcloud extends Provider {
  static providerName = "ovh";
  static baseUrl = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1";
  static keyless = true;
  // 2/min is per model; key-level pacing stays loose and each model is benched
  // on its own 429 (rateLimitScope below).
  static tiers: Record<string, TierLimit> = { free: { rpm: 10, rpd: null } };
  // From the public catalog, 2026-10-09; discovery replaces the list.
  static defaultModels: ModelSpec[] = [
    modelSpec("Meta-Llama-3_3-70B-Instruct", ["chat", "large"]),
    modelSpec("Qwen3.8-27B", ["chat"]),
    modelSpec("Mistral-Small-3.2-24B-Instruct-2506", ["chat"]),
    modelSpec("gpt-oss-120b", ["chat", "large", "reasoning"]),
    modelSpec("gpt-oss-20b", ["chat", "small", "fast", "reasoning"]),
    modelSpec("Qwen3.5-9B", ["chat", "small", "fast"]),
  ];

  /** Keys mean paid usage here, so none are accepted: anonymous only. */
  constructor(opts: ProviderOptions = {}) {
    super(ANONYMOUS, { discover: !opts.models, ...opts });
  }

  rateLimitScope(_body: string): ErrorScope {
    return "model"; // the anonymous limit is per IP *per model*
  }
}
