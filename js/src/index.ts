export { FreeLLM } from "./client.js";
export type { FreeLLMOptions, ChatOptions } from "./client.js";
export { Provider, OpenRouter, GoogleAIStudio, Gemini, NIM, Groq, Cerebras, Mistral, Kilo, OVHcloud, ZAI, Cohere, CloudflareWorkersAI } from "./providers/index.js";
export type { ProviderOptions, TierLimit, CloudflareOptions } from "./providers/index.js";
export { providersFromEnv, PROVIDER_ENV, envKeys, envVars, buildProvider, KEYLESS, keylessMode } from "./config.js";
export type { ProviderEnv, KeylessArg } from "./config.js";
export { listFreeModels, toSpecs, discover } from "./discovery.js";
export { modelSpec, resolveModels, isVirtual } from "./registry.js";
export type { ModelSpec } from "./registry.js";
export { ChatResponse } from "./types.js";
export type { Message, Choice, Usage, MessageLike, ChatRequest, FreeLLMEvent } from "./types.js";
export { StateStore } from "./state.js";
export { serve, createServer } from "./server.js";
export type { ServeOptions } from "./server.js";
export {
  FreeLLMError,
  ConfigError,
  ProviderError,
  AuthError,
  BadRequest,
  QuotaExhausted,
  RateLimited,
  Transient,
  ModelNotFound,
  NoProvidersAvailable,
  classify,
  parseRetryAfter,
} from "./errors.js";
export type { ErrorScope, RateScope } from "./errors.js";

export { VERSION } from "./version.js";
