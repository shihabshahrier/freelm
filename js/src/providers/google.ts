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
  // Gemini 1.5 is retired for new projects; 2.5 flash family is the current
  // free-tier workhorse, with 2.0 flash kept as a fallback (2026-06).
  // 2.5-flash *thinks by default* and can spend a small max_tokens budget
  // entirely on reasoning (empty text, finish_reason=length) — so the
  // non-thinking lite leads for `auto`, and flash is tagged "reasoning".
  static defaultModels: ModelSpec[] = [
    modelSpec("gemini-2.5-flash-lite", ["chat", "fast", "small"], 1000000),
    modelSpec("gemini-2.5-flash", ["chat", "fast", "large", "reasoning"], 1000000),
    modelSpec("gemini-2.0-flash", ["chat", "fast"], 1000000),
  ];
}

export { GoogleAIStudio as Gemini };
