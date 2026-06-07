/** Build providers from environment variables. */
import { ConfigError } from "./errors.js";
import { Cerebras, GoogleAIStudio, Groq, Mistral, NIM, OpenRouter, Provider } from "./providers/index.js";

function split(value: string | undefined): string[] {
  return value ? value.split(",").map((s) => s.trim()).filter(Boolean) : [];
}

function firstEnv(...names: string[]): string | undefined {
  for (const n of names) {
    const v = process.env[n];
    if (v) return v;
  }
  return undefined;
}

export function providersFromEnv(): Provider[] {
  const provs: Provider[] = [];

  const ork = split(firstEnv("OPENROUTER_API_KEY", "FREELM_OPENROUTER_KEYS"));
  if (ork.length) provs.push(new OpenRouter(ork, { tier: process.env.FREELM_OPENROUTER_TIER ?? "free" }));

  const gk = split(firstEnv("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_KEY", "FREELM_GOOGLE_KEYS"));
  if (gk.length) provs.push(new GoogleAIStudio(gk, { tier: process.env.FREELM_GOOGLE_TIER ?? "free" }));

  const nk = split(firstEnv("NVIDIA_API_KEY", "NIM_API_KEY", "FREELM_NIM_KEYS"));
  if (nk.length) provs.push(new NIM(nk, { tier: process.env.FREELM_NIM_TIER ?? "free" }));

  const groqKeys = split(firstEnv("GROQ_API_KEY", "FREELM_GROQ_KEYS"));
  if (groqKeys.length) provs.push(new Groq(groqKeys, { tier: process.env.FREELM_GROQ_TIER ?? "free" }));

  const ck = split(firstEnv("CEREBRAS_API_KEY", "FREELM_CEREBRAS_KEYS"));
  if (ck.length) provs.push(new Cerebras(ck, { tier: process.env.FREELM_CEREBRAS_TIER ?? "free" }));

  const mk = split(firstEnv("MISTRAL_API_KEY", "FREELM_MISTRAL_KEYS"));
  if (mk.length) provs.push(new Mistral(mk, { tier: process.env.FREELM_MISTRAL_TIER ?? "free" }));

  if (!provs.length) {
    throw new ConfigError(
      "no provider keys found in environment. Set at least one of OPENROUTER_API_KEY, " +
        "GEMINI_API_KEY, NVIDIA_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY, or MISTRAL_API_KEY.",
    );
  }
  return provs;
}
