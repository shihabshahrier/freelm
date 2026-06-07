import { modelSpec, ModelSpec } from "../registry.js";
import { Provider, TierLimit } from "./base.js";

/** NVIDIA NIM — OpenAI-compatible, free against build credits. */
export class NIM extends Provider {
  static providerName = "nim";
  static baseUrl = "https://integrate.api.nvidia.com/v1";
  static tiers: Record<string, TierLimit> = {
    free: { rpm: 40, rpd: null },
  };
  static defaultModels: ModelSpec[] = [
    modelSpec("meta/llama-3.3-70b-instruct", ["chat", "large"], 128000),
    modelSpec("meta/llama-3.1-70b-instruct", ["chat", "large"], 128000),
    modelSpec("meta/llama-3.1-8b-instruct", ["chat", "small", "fast"], 128000),
  ];
}
