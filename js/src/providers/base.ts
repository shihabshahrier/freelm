/** Base provider. All shipped providers speak OpenAI-compatible HTTP, so the base
 * does request shaping + response parsing; subclasses declare endpoint, auth,
 * default models, and tier limits as STATIC members (read via this.constructor,
 * which sidesteps JS class-field init ordering). Providers are pure (no I/O). */
import { ConfigError } from "../errors.js";
import { KeyState, newKeyState } from "../keys.js";
import { ModelSpec, resolveModels } from "../registry.js";
import { ChatResponse, Choice, usageFrom } from "../types.js";

export interface TierLimit {
  rpm: number | null;
  rpd: number | null;
}

export interface ProviderOptions {
  tier?: string;
  models?: ModelSpec[];
  rpm?: number | null;
  rpd?: number | null;
  priority?: number;
  name?: string;
  baseUrl?: string;
  extraHeaders?: Record<string, string>;
  discover?: boolean;
  discoverFreeOnly?: boolean;
  cacheTtl?: number | null;
}

export class Provider {
  static providerName = "base";
  static baseUrl = "";
  static chatPath = "/chat/completions";
  static modelsPath = "/models";
  static tiers: Record<string, TierLimit> = { free: { rpm: 20, rpd: null } };
  static defaultModels: ModelSpec[] = [];

  name: string;
  baseUrl: string;
  chatPath: string;
  modelsPath: string;
  tier: string;
  rpm: number | null;
  rpd: number | null;
  priority: number;
  extraHeaders: Record<string, string>;
  discover: boolean;
  discoverFreeOnly: boolean;
  cacheTtl: number | null;
  models: ModelSpec[];
  keys: KeyState[];
  _rr = 0;
  _discovered = false;

  constructor(keys: string | string[], opts: ProviderOptions = {}) {
    const cls = this.constructor as typeof Provider;
    const keyList = (Array.isArray(keys) ? keys : [keys]).map((k) => (k ?? "").trim()).filter(Boolean);
    this.name = opts.name ?? cls.providerName;
    this.baseUrl = opts.baseUrl ?? cls.baseUrl;
    this.chatPath = cls.chatPath;
    this.modelsPath = cls.modelsPath;
    if (!keyList.length) throw new ConfigError(`${this.name}: no API keys provided`);

    this.tier = opts.tier ?? "free";
    const tdef = cls.tiers[this.tier] ?? {};
    this.rpm = opts.rpm !== undefined ? opts.rpm : tdef.rpm ?? 20;
    this.rpd = opts.rpd !== undefined ? opts.rpd : tdef.rpd ?? null;
    this.priority = opts.priority ?? 0;
    this.extraHeaders = { ...(opts.extraHeaders ?? {}) };
    this.discover = opts.discover ?? false;
    this.discoverFreeOnly = opts.discoverFreeOnly ?? false;
    this.cacheTtl = opts.cacheTtl ?? null;
    this.models = opts.models ? [...opts.models] : [...cls.defaultModels];
    this.keys = keyList.map((k) => newKeyState(k, this.tier, this.rpm, this.rpd));
  }

  get url(): string {
    return this.baseUrl.replace(/\/+$/, "") + this.chatPath;
  }

  discoveryUrl(): string {
    return this.baseUrl.replace(/\/+$/, "") + this.modelsPath;
  }

  authHeaders(key: string): Record<string, string> {
    return { Authorization: `Bearer ${key}` };
  }

  headers(key: string): Record<string, string> {
    return { "Content-Type": "application/json", ...this.authHeaders(key), ...this.extraHeaders };
  }

  resolveModels(alias: string): string[] {
    return resolveModels(this.models, alias);
  }

  /** Is a 429 account/key-wide ("key") or just this model ("model")? */
  rateLimitScope(_body: string): "key" | "model" {
    return "key";
  }

  parseResponse(data: any, latencyMs: number): ChatResponse {
    const choices: Choice[] = (data.choices ?? []).map((c: any) => ({
      index: c.index ?? 0,
      message: {
        role: c.message?.role ?? "assistant",
        content: c.message?.content ?? null,
        tool_calls: c.message?.tool_calls ?? null,
      },
      finish_reason: c.finish_reason ?? null,
    }));
    return new ChatResponse(data.id ?? null, data.model ?? null, this.name, choices, usageFrom(data.usage), latencyMs, data);
  }

  capacity(now: number): number {
    return this.keys.reduce((s, k) => s + k.remaining(now), 0);
  }

  avgLatency(): number {
    const v = this.keys.map((k) => k.ewmaLatency).filter((x) => x > 0);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : Infinity;
  }
}
