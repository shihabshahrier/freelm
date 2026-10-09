import { ErrorScope } from "../errors.js";
import { modelSpec, ModelSpec } from "../registry.js";
import { Provider, TierLimit } from "./base.js";

/** Google AI Studio (Gemini) via its OpenAI-compatible endpoint. */
export class GoogleAIStudio extends Provider {
  static providerName = "google";
  static baseUrl = "https://generativelanguage.googleapis.com/v1beta/openai";
  static tiers: Record<string, TierLimit> = {
    free: { rpm: 15, rpd: 1500 },
    tier1: { rpm: 2000, rpd: null },
  };
  // Verified live on the free tier 2026-10-09. gemini-2.0-flash and
  // gemini-2.5-pro are retired (404); Pro/preview "omni"/deep-research models
  // have no free quota (429, limit 0), so they're left out. Thinking models
  // (2.5-flash, 3.x-flash) can spend a small max_tokens budget entirely on
  // reasoning (empty text, finish_reason=length) — the non-thinking lite
  // models lead for `auto`, and the thinkers are tagged "reasoning".
  static defaultModels: ModelSpec[] = [
    modelSpec("gemini-2.5-flash-lite", ["chat", "fast", "small", "tools", "vision"], 1048576),
    modelSpec("gemini-3.1-flash-lite", ["chat", "fast", "small", "tools", "vision"], 1048576),
    modelSpec("gemini-2.5-flash", ["chat", "fast", "large", "tools", "vision", "reasoning"], 1048576),
    modelSpec("gemini-3-flash-preview", ["chat", "large", "tools", "vision", "reasoning"], 1048576),
    modelSpec("gemini-flash-lite-latest", ["chat", "fast", "small", "tools", "vision"], 1048576),
  ];

  /** AI Studio quotas (RPM/TPM/RPD) are per model, so a 429 on one model
   * leaves the others usable on the same key. */
  rateLimitScope(_body: string): ErrorScope {
    return "model";
  }

  /** "This model is currently experiencing high demand" (503) is about one
   * model; other Gemini models on the same key keep working. */
  transientScope(body: string): ErrorScope {
    const b = (body || "").toLowerCase();
    return b.includes("high demand") || b.includes("overloaded") ? "model" : "key";
  }
}

export { GoogleAIStudio as Gemini };
