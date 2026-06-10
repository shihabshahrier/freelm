/** Dynamic model discovery via the OpenAI-compatible GET /models endpoint.
 * Resolution order: live API -> disk cache -> hardcoded defaults. */
import * as cache from "./cache.js";
import { ModelSpec, modelSpec } from "./registry.js";

const NON_CHAT = [
  "whisper", "tts", "text-to-speech", "speech", "audio", "transcribe",
  "embed", "embedding", "rerank", "moderation", "guard", "ocr", "-vision-encoder",
  "imagen", "veo", "image-generation", "-generate-", "stable-diffusion", "dall-e", "aqa",
  "orpheus", "playai", "sonic", "voice", "-stt", "-asr",
];
const LARGE_HINTS = ["ultra", "super", "-405", "235b", "120b", "-large", "-xl"];
const SMALL_HINTS = ["mini", "nano", "small", "lite", "tiny", "-xs", "edge"];
const REASONING_HINTS = ["gpt-oss", "deepseek-r1", "magistral", "qwq", "thinking", "-think", "reasoning"];

function paramsB(id: string): number {
  const nums = [...id.toLowerCase().matchAll(/(\d+(?:\.\d+)?)\s*b\b/g)].map((m) => parseFloat(m[1]));
  return nums.length ? Math.max(...nums) : 0;
}

function sizeTags(id: string): string[] {
  const s = id.toLowerCase();
  const smallKw = SMALL_HINTS.some((h) => s.includes(h));
  const largeKw = LARGE_HINTS.some((h) => s.includes(h));
  if (smallKw && !largeKw) return ["small", "fast"];
  if (largeKw && !smallKw) return ["large"];
  const big = paramsB(s);
  if (big > 0) {
    if (big >= 30) return ["large"];
    if (big <= 20) return ["small", "fast"];
  }
  return [];
}

export function toSpecs(apiModels: any[], freeOnly: boolean): ModelSpec[] {
  const specs: ModelSpec[] = [];
  for (const m of apiModels) {
    const mid: string | undefined = m?.id;
    if (!mid) continue;
    if (freeOnly && !mid.endsWith(":free")) continue;
    const low = mid.toLowerCase();
    if (NON_CHAT.some((t) => low.includes(t))) continue;

    const arch = m.architecture || {};
    const outMod: string[] = m.output_modalities || arch.output_modalities || ["text"];
    if (!outMod.includes("text")) continue;

    const ctx = m.context_length || m.top_provider?.context_length || 0;
    const params = (m.supported_parameters || []).map((p: any) => String(p).toLowerCase());
    const inMod: string[] = arch.input_modalities || [];

    const tags = ["chat", ...sizeTags(mid)];
    if (params.includes("tools") || params.includes("tool_choice")) tags.push("tools");
    if (params.includes("reasoning") || params.includes("include_reasoning") || REASONING_HINTS.some((h) => low.includes(h))) {
      tags.push("reasoning");
    }
    if (inMod.includes("image") || params.includes("vision")) tags.push("vision");

    specs.push(modelSpec(mid, [...new Set(tags)], Math.trunc(ctx) || 0));
  }

  // `auto` order: capable but fast/predictable. Giant (>150B) and reasoning models
  // rank after plain instruct models; then prefer large, then bigger context.
  specs.sort((a, b) => {
    const ga = paramsB(a.id) > 150 ? 1 : 0, gb = paramsB(b.id) > 150 ? 1 : 0;
    if (ga !== gb) return ga - gb;
    const ra = a.tags.includes("reasoning") ? 1 : 0, rb = b.tags.includes("reasoning") ? 1 : 0;
    if (ra !== rb) return ra - rb;
    const la = a.tags.includes("large") ? 0 : 1, lb = b.tags.includes("large") ? 0 : 1;
    if (la !== lb) return la - lb;
    return b.ctx - a.ctx;
  });
  return specs;
}

function rawModels(payload: any): any[] {
  return payload?.data || payload?.models || [];
}

function apply(provider: any, raw: any[]): boolean {
  const specs = toSpecs(raw, provider.discoverFreeOnly ?? false);
  if (specs.length) {
    provider.models = specs;
    provider._discovered = true;
    return true;
  }
  return false;
}

/** Populate provider.models from the live API (or cache). Never throws — on
 * failure the provider keeps its hardcoded fallback models. */
export async function discover(provider: any): Promise<boolean> {
  const cached = cache.load(provider.name);
  // a cached list that yields no usable specs falls through to a live fetch
  if (cached && apply(provider, cached)) return true;
  try {
    const res = await fetch(provider.discoveryUrl(), {
      headers: provider.headers(provider.keys[0].key),
      signal: AbortSignal.timeout(15_000), // a stalled /models must not hang the first chat()
    });
    if (res.status === 200) {
      const raw = rawModels(await res.json());
      if (raw.length) {
        cache.save(provider.name, raw, provider.cacheTtl ?? null);
        return apply(provider, raw);
      }
    }
  } catch {
    // keep fallback
  }
  return false;
}

/** Discover OpenRouter free models without building a client. */
export async function listFreeModels(apiKey?: string, refresh = false): Promise<ModelSpec[]> {
  const { OpenRouter } = await import("./providers/openrouter.js");
  const key = apiKey || process.env.OPENROUTER_API_KEY || "none";
  if (refresh) cache.clear("openrouter");
  const p = new OpenRouter(key);
  await discover(p);
  return p.models;
}
