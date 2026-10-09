import { modelSpec, ModelSpec } from "../registry.js";
import { Provider, TierLimit } from "./base.js";

/** NVIDIA NIM — OpenAI-compatible, free against build.nvidia.com credits. */
export class NIM extends Provider {
  static providerName = "nim";
  static baseUrl = "https://integrate.api.nvidia.com/v1";
  static tiers: Record<string, TierLimit> = {
    free: { rpm: 40, rpd: null },
  };
  // meta/llama-3.x reached end of life 2026-08-26 (HTTP 410). These ids are
  // in the live catalog as of 2026-10-09; retired ones get benched at runtime.
  static defaultModels: ModelSpec[] = [
    modelSpec("nvidia/nemotron-3-super-120b-a12b", ["chat", "large", "tools", "reasoning"]),
    modelSpec("deepseek-ai/deepseek-v4.1-flash", ["chat", "large", "tools"]),
    modelSpec("z-ai/glm-5.3-flash", ["chat", "fast", "tools"]),
    modelSpec("moonshotai/kimi-k2.6", ["chat", "large", "tools"]),
    modelSpec("nvidia/nemotron-3.5-lightning-30b-a3b", ["chat", "small", "fast"]),
    modelSpec("openai/gpt-oss-20b", ["chat", "small", "fast", "tools", "reasoning"]),
  ];
}
