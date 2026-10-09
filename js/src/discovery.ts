/** Dynamic model discovery via the OpenAI-compatible GET /models endpoint.
 * Resolution order: live API -> disk cache -> hardcoded defaults. */
import * as cache from "./cache.js";
import { ModelSpec, modelSpec } from "./registry.js";
import { env } from "./runtime.js";

const NON_CHAT = [
  "whisper", "tts", "text-to-speech", "speech", "audio", "transcribe",
  "embed", "embedding", "rerank", "moderation", "guard", "ocr", "-vision-encoder",
  "imagen", "veo", "image-generation", "-generate-", "stable-diffusion", "dall-e", "aqa",
  "orpheus", "playai", "sonic", "voice", "-stt", "-asr",
  // classifiers / safety filters / reward models answer with labels, not chat
  "safety", "content-safety", "topic-control", "reward", "-parse", "nemoretriever", "nvclip", "bge",
];
// size words are matched as whole tokens of the id ("gemini" must not read as "mini")
const LARGE_HINTS = new Set(["ultra", "super", "large", "xl", "405b", "235b", "120b"]);
const SMALL_HINTS = new Set(["mini", "nano", "small", "lite", "tiny", "xs", "edge"]);
const REASONING_HINTS = ["gpt-oss", "deepseek-r1", "magistral", "qwq", "thinking", "-think", "reasoning"];
/** A catalog fetch must never eat the caller's whole request budget. */
const DISCOVERY_TIMEOUT_MS = 10_000;

function paramsB(id: string): number {
  const nums = [...id.toLowerCase().matchAll(/(\d+(?:\.\d+)?)\s*b\b/g)].map((m) => parseFloat(m[1]));
  return nums.length ? Math.max(...nums) : 0;
}

export function sizeTags(id: string): string[] {
  const s = id.toLowerCase();
  const tokens = new Set(s.split(/[^a-z0-9.]+/));
  const smallKw = [...tokens].some((t) => SMALL_HINTS.has(t));
  const largeKw = [...tokens].some((t) => LARGE_HINTS.has(t));
  // explicit keyword (nano/mini/ultra/...) wins over raw parameter count,
  // which can be misleading (e.g. "nano-30b" is meant to be the small one).
  if (smallKw && !largeKw) return ["small", "fast"];
  if (largeKw && !smallKw) return ["large"];
  const big = paramsB(s);
  if (big > 0) {
    if (big >= 30) return ["large"];
    if (big <= 20) return ["small", "fast"];
  }
  return [];
}

/** `:free` ids, entries flagged `isFree` (Kilo) and zero-priced entries are
 * free. Catalogs without pricing (Groq, Mistral, NIM, ...) belong to free-tier
 * accounts, so they count as free — except under `strict` (filtering a mixed
 * paid/free catalog), where only positive evidence does. */
function isFree(mid: string, m: any, strict = false): boolean {
  if (mid.endsWith(":free")) return true;
  if (typeof m?.isFree === "boolean") return m.isFree;
  const pricing = m?.pricing;
  if (!pricing || typeof pricing !== "object" || !Object.keys(pricing).length) return !strict;
  // every listed price must be zero (image/audio/request fees included)
  return Object.values(pricing).every((v) => {
    if (v === null || v === undefined || typeof v === "boolean") return true;
    const n = Number(v);
    return Number.isFinite(n) && n === 0;
  });
}

export function toSpecs(apiModels: any[], freeOnly: boolean): ModelSpec[] {
  const specs: ModelSpec[] = [];
  for (const m of apiModels) {
    let mid: unknown = m?.id;
    if (!mid || typeof mid !== "string") continue;
    if (mid.startsWith("models/")) mid = mid.slice("models/".length); // Google lists "models/gemini-..."
    const id = mid as string;
    if (freeOnly && !isFree(id, m, true)) continue; // paid entry in a mixed catalog (OpenRouter, Kilo)
    const low = id.toLowerCase();
    if (NON_CHAT.some((t) => low.includes(t))) continue;

    const arch = m.architecture || {};
    const outMod: string[] = m.output_modalities || arch.output_modalities || ["text"];
    if (outMod.length !== 1 || outMod[0] !== "text") continue; // audio/image/music generators aren't chat models
    // Mistral: {"capabilities": {"completion_chat": true, "function_calling": true, "vision": false}}
    const caps = m.capabilities && typeof m.capabilities === "object" ? m.capabilities : {};
    if (caps.completion_chat === false) continue;

    const rawCtx = m.context_length || m.context_window || m.max_context_length || m.top_provider?.context_length || 0;
    const ctx = Math.trunc(Number(rawCtx)) || 0;
    const params = (m.supported_parameters || []).map((p: any) => String(p).toLowerCase());
    const inMod: string[] = arch.input_modalities || [];

    const tags = ["chat", ...sizeTags(id)];
    if (params.includes("tools") || params.includes("tool_choice") || caps.function_calling) tags.push("tools");
    if (params.includes("reasoning") || params.includes("include_reasoning") || REASONING_HINTS.some((h) => low.includes(h))) {
      tags.push("reasoning");
    }
    if (inMod.includes("image") || params.includes("vision") || caps.vision) tags.push("vision");

    specs.push(modelSpec(id, [...new Set(tags)], ctx, isFree(id, m)));
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
  const d = payload?.data || payload?.models || [];
  return Array.isArray(d) ? d : [];
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

/** Keys to try for GET /models: usable ones first, at most three. A dead first
 * key (401) must not hide a healthy second key's catalog. */
function discoveryKeys(provider: any): string[] {
  const usable = provider.keys.filter((k: any) => !k.disabled);
  return (usable.length ? usable : provider.keys).slice(0, 3).map((k: any) => k.key);
}

/** Populate provider.models from the live API (or cache). Never throws — on
 * failure the provider keeps its hardcoded fallback models. */
export async function discover(provider: any): Promise<boolean> {
  try {
    const cached = cache.load(provider.name);
    // a cached list that yields no usable specs falls through to a live fetch
    if (cached && apply(provider, cached)) return true;
    for (const key of discoveryKeys(provider)) {
      let res: Response;
      try {
        res = await fetch(provider.discoveryUrl(), {
          headers: provider.headers(key),
          signal: AbortSignal.timeout(DISCOVERY_TIMEOUT_MS), // a stalled /models must not hang the first chat()
        });
      } catch {
        return false; // network trouble: don't hammer it once per key
      }
      if (res.status !== 200) continue; // e.g. this key is dead — try the next one
      const raw = rawModels(await res.json());
      if (!raw.length) return false;
      cache.save(provider.name, raw, provider.cacheTtl ?? null);
      return apply(provider, raw);
    }
  } catch {
    // keep fallback
  }
  return false;
}

/** Discover OpenRouter free models without building a client. */
export async function listFreeModels(apiKey?: string, refresh = false): Promise<ModelSpec[]> {
  const { OpenRouter } = await import("./providers/openrouter.js");
  const key = apiKey || env("OPENROUTER_API_KEY") || "none";
  if (refresh) cache.clear("openrouter");
  const p = new OpenRouter(key);
  await discover(p);
  return p.models;
}
