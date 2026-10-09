import { ConfigError, ErrorScope } from "../errors.js";
import { modelSpec, ModelSpec } from "../registry.js";
import { Provider, ProviderOptions, TierLimit } from "./base.js";

const ACCOUNT_ID = /^[0-9a-f]{32}$/;

export interface CloudflareOptions extends ProviderOptions {
  /** The 32-character account id from the Cloudflare dashboard. */
  accountId?: string | null;
}

/** Cloudflare Workers AI via its OpenAI-compatible endpoint. Needs an API token
 * with Workers AI permission **and** the account id (both on the Cloudflare
 * dashboard).
 *
 * Every account gets 10,000 Neurons/day free (a few hundred chats; checked
 * 2026-10). On the Free plan requests fail past that — freelm then rests the key
 * and re-checks hourly; on the Workers Paid plan the overage is billed, so keep
 * a Free-plan account for freelm. Some newer models (Kimi K2.6, DeepSeek V4,
 * GLM-5) need Workers Paid and aren't listed. There is no OpenAI-style model
 * list, so the models below are curated. One provider per account: add a second
 * as `new CloudflareWorkersAI(token, { accountId, name: "cloudflare-2" })`. */
export class CloudflareWorkersAI extends Provider {
  static providerName = "cloudflare";
  static baseUrl = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1";
  // text generation: 300 requests/minute (checked 2026-10); the daily Neuron
  // allowance is the real limit
  static tiers: Record<string, TierLimit> = { free: { rpm: 300, rpd: null } };
  // Free-plan models on 2026-10-09 (workers-ai/models), non-thinking first.
  static defaultModels: ModelSpec[] = [
    modelSpec("@cf/meta/llama-4-scout-17b-16e-instruct", ["chat", "tools", "vision"], 131000),
    modelSpec("@cf/mistralai/mistral-small-3.1-24b-instruct", ["chat", "tools"], 128000),
    modelSpec("@cf/openai/gpt-oss-120b", ["chat", "large", "tools", "reasoning"], 128000),
    modelSpec("@cf/qwen/qwen3.8-27b", ["chat", "tools", "vision", "reasoning"], 262144),
    modelSpec("@cf/google/gemma-4-26b-a4b-it", ["chat", "tools", "vision", "reasoning"], 256000),
    modelSpec("@cf/zai-org/glm-4.7-flash", ["chat", "fast", "tools", "reasoning"], 131072),
    modelSpec("@cf/meta/llama-3.3-70b-instruct-fp8-fast", ["chat", "large"], 24000),
    modelSpec("@cf/openai/gpt-oss-20b", ["chat", "small", "fast", "tools", "reasoning"], 128000),
    modelSpec("@cf/meta/llama-3.1-8b-instruct-fp8", ["chat", "small", "fast"], 32000),
  ];
  /** Many models (Llama, gpt-oss, Mistral, Granite) default to 256 output
   * tokens, silently cutting answers short — send this unless the caller sets one. */
  static defaultMaxTokens = 4096;

  accountId: string | null;

  constructor(keys: string | string[], opts: CloudflareOptions = {}) {
    const { accountId: raw, ...rest } = opts;
    const accountId = (raw ?? "").trim().toLowerCase() || null;
    if (accountId !== null && !ACCOUNT_ID.test(accountId)) {
      throw new ConfigError("cloudflare: accountId must be the 32-character hex id from your Cloudflare dashboard");
    }
    if (!rest.baseUrl) {
      if (accountId === null) {
        throw new ConfigError(
          "cloudflare: accountId is required — set CLOUDFLARE_ACCOUNT_ID (the 32-character id on your Cloudflare dashboard)",
        );
      }
      rest.baseUrl = (new.target as typeof CloudflareWorkersAI).baseUrl.replace("{account_id}", accountId);
    }
    super(keys, { discover: false, ...rest }); // no OpenAI-style model list
    this.accountId = accountId;
  }

  adaptPayload(body: Record<string, any>): Record<string, any> {
    if (body.max_tokens == null && body.max_completion_tokens == null) {
      return { ...body, max_tokens: (this.constructor as typeof CloudflareWorkersAI).defaultMaxTokens };
    }
    return body;
  }

  /** 3040 "Capacity temporarily exceeded" is one model; the daily Neuron
   * allowance (4006/3036) and the per-minute limit are the account's. */
  rateLimitScope(body: string): ErrorScope {
    return (body ?? "").toLowerCase().includes("capacity") ? "model" : "key";
  }
}
