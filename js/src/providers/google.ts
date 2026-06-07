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
  static defaultModels: ModelSpec[] = [
    modelSpec("gemini-2.0-flash", ["chat", "fast", "large"], 1000000),
    modelSpec("gemini-2.0-flash-lite", ["chat", "fast", "small"], 1000000),
    modelSpec("gemini-1.5-flash", ["chat", "fast"], 1000000),
  ];
}

export { GoogleAIStudio as Gemini };
