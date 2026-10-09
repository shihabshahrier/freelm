/** Build providers from environment variables (comma-separate several keys).
 *
 * Keyless public endpoints (Kilo Gateway, OVHcloud) need no signup at all. They
 * are added only when asked: `keyless: true` / FREELM_KEYLESS=1 (always, as
 * last-resort fallbacks) or "auto" (only when no keys are configured — what the
 * CLI uses). The library never routes prompts to them silently. */
import { ConfigError } from "./errors.js";
import { Cerebras, GoogleAIStudio, Groq, Kilo, Mistral, NIM, OpenRouter, OVHcloud, Provider } from "./providers/index.js";
import { env } from "./runtime.js";

/** How one provider is configured from the environment. */
export interface ProviderEnv {
  name: string;
  cls: new (keys: string[], opts?: any) => Provider;
  keyVars: string[]; // first non-empty wins; comma-separated = several keys
  tierVar: string;
  signupUrl: string; // where to get a free key
}

// Order = default provider order for fromEnv().
// NB: Groq (gsk_...) is the free provider here; xAI Grok (xai-...) is a
// different, paid service and is intentionally not supported.
export const PROVIDER_ENV: ProviderEnv[] = [
  { name: "openrouter", cls: OpenRouter, keyVars: ["OPENROUTER_API_KEY", "FREELM_OPENROUTER_KEYS"],
    tierVar: "FREELM_OPENROUTER_TIER", signupUrl: "https://openrouter.ai/keys" },
  { name: "google", cls: GoogleAIStudio, keyVars: ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_KEY", "FREELM_GOOGLE_KEYS"],
    tierVar: "FREELM_GOOGLE_TIER", signupUrl: "https://aistudio.google.com/apikey" },
  { name: "nim", cls: NIM, keyVars: ["NVIDIA_API_KEY", "NIM_API_KEY", "FREELM_NIM_KEYS"],
    tierVar: "FREELM_NIM_TIER", signupUrl: "https://build.nvidia.com/settings/api-keys" },
  { name: "groq", cls: Groq, keyVars: ["GROQ_API_KEY", "FREELM_GROQ_KEYS"],
    tierVar: "FREELM_GROQ_TIER", signupUrl: "https://console.groq.com/keys" },
  { name: "cerebras", cls: Cerebras, keyVars: ["CEREBRAS_API_KEY", "FREELM_CEREBRAS_KEYS"],
    tierVar: "FREELM_CEREBRAS_TIER", signupUrl: "https://cloud.cerebras.ai" },
  { name: "mistral", cls: Mistral, keyVars: ["MISTRAL_API_KEY", "FREELM_MISTRAL_KEYS"],
    tierVar: "FREELM_MISTRAL_TIER", signupUrl: "https://console.mistral.ai/api-keys" },
  { name: "kilo", cls: Kilo, keyVars: ["KILO_API_KEY", "FREELM_KILO_KEYS"],
    tierVar: "FREELM_KILO_TIER", signupUrl: "https://app.kilo.ai" },
];

/** No-signup endpoints, tried last (priority 100) when keyless mode is on. */
export const KEYLESS: Array<new (...args: any[]) => Provider> = [Kilo, OVHcloud];
export const KEYLESS_NOTICE =
  "freelm: no API keys found — using keyless public endpoints (Kilo Gateway, OVHcloud). " +
  "Limits are low and free routes may log prompts; add a free key for more: `freelm doctor`.";
export type KeylessArg = boolean | "auto" | "always" | "never" | string | null | undefined;

/** "always" | "auto" | "never" from an argument or FREELM_KEYLESS (library default: never). */
export function keylessMode(value?: KeylessArg): "always" | "auto" | "never" {
  if (value === true) return "always";
  if (value === false) return "never";
  const v = String(value ?? env("FREELM_KEYLESS") ?? "").trim().toLowerCase();
  if (["1", "true", "yes", "on", "always"].includes(v)) return "always";
  if (v === "auto") return "auto";
  return "never";
}

function split(value: string | undefined): string[] {
  return value ? value.split(",").map((s) => s.trim()).filter(Boolean) : [];
}

function firstEnv(...names: string[]): string | undefined {
  for (const n of names) {
    const v = env(n);
    if (v) return v;
  }
  return undefined;
}

/** The keys configured for one provider (empty if none). */
export function envKeys(spec: ProviderEnv): string[] {
  return split(firstEnv(...spec.keyVars));
}

/** Providers for every key found in the environment, plus the keyless
 * endpoints when `keyless` (or FREELM_KEYLESS) asks for them. */
export function providersFromEnv(keyless?: KeylessArg): Provider[] {
  const provs: Provider[] = [];
  for (const spec of PROVIDER_ENV) {
    const keys = envKeys(spec);
    if (keys.length) provs.push(new spec.cls(keys, { tier: env(spec.tierVar) ?? "free" }));
  }
  const mode = keylessMode(keyless);
  if (mode === "always" || (mode === "auto" && !provs.length)) {
    const keyed = provs.length > 0;
    const have = new Set(provs.map((p) => p.name));
    for (const cls of KEYLESS) {
      const p = cls === OVHcloud ? new OVHcloud({ priority: 100 }) : new (cls as any)(null, { priority: 100 });
      if (!have.has(p.name)) provs.push(p);
    }
    if (!keyed) (globalThis as any).process?.stderr?.write?.(KEYLESS_NOTICE + "\n");
  }
  if (!provs.length) {
    throw new ConfigError(
      "no provider keys found in environment. Set at least one of " +
        PROVIDER_ENV.filter((s) => s.cls !== Kilo).map((s) => s.keyVars[0]).join(", ") +
        " (free keys: run `freelm doctor` for signup links), or set FREELM_KEYLESS=1 to use " +
        "keyless public endpoints (Kilo Gateway, OVHcloud).",
    );
  }
  return provs;
}
