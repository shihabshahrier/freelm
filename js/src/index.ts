export { FreeLLM } from "./client.js";
export type { FreeLLMOptions, ChatOptions } from "./client.js";
export { Provider, OpenRouter, GoogleAIStudio, Gemini, NIM, Groq, Cerebras, Mistral } from "./providers/index.js";
export type { ProviderOptions, TierLimit } from "./providers/index.js";
export { providersFromEnv } from "./config.js";
export { listFreeModels, toSpecs, discover } from "./discovery.js";
export { modelSpec, resolveModels } from "./registry.js";
export type { ModelSpec } from "./registry.js";
export { ChatResponse } from "./types.js";
export type { Message, Choice, Usage, MessageLike, ChatRequest } from "./types.js";
export {
  FreeLLMError,
  ConfigError,
  ProviderError,
  AuthError,
  RateLimited,
  Transient,
  ModelNotFound,
  NoProvidersAvailable,
} from "./errors.js";

export const VERSION = "0.1.0";
