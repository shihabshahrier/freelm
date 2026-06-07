/** Candidate ordering strategies. A candidate is one (provider, key, model). */
import type { KeyState } from "./keys.js";

export const STRATEGIES = ["priority", "round_robin", "quota_aware", "latency"] as const;
export type Strategy = (typeof STRATEGIES)[number];

export interface Candidate {
  provider: any;
  key: KeyState;
  model: string;
}

export function orderCandidates(
  providers: any[],
  alias: string,
  now: number,
  strategy: string,
  rr: { p: number },
): Candidate[] {
  let provs = [...providers];

  if (strategy === "round_robin" && provs.length) {
    const i = (rr.p ?? 0) % provs.length;
    provs = [...provs.slice(i), ...provs.slice(0, i)];
    rr.p = (rr.p ?? 0) + 1;
  } else if (strategy === "quota_aware") {
    provs.sort((a, b) => b.capacity(now) - a.capacity(now));
  } else if (strategy === "latency") {
    provs.sort((a, b) => a.avgLatency() - b.avgLatency());
  } else {
    provs.sort((a, b) => a.priority - b.priority);
  }

  // Build each provider's own ordered sublist (rotated keys, then models).
  const perProvider: Candidate[][] = [];
  for (const p of provs) {
    let keys = [...p.keys];
    if (keys.length) {
      const ki = p._rr % keys.length;
      keys = [...keys.slice(ki), ...keys.slice(0, ki)];
      p._rr++;
    }
    const models = p.resolveModels(alias);
    const sub: Candidate[] = [];
    for (const k of keys) for (const mid of models) sub.push({ provider: p, key: k, model: mid });
    if (sub.length) perProvider.push(sub);
  }

  // Interleave breadth-first across providers (best model of each, then next).
  const out: Candidate[] = [];
  const maxLen = perProvider.reduce((m, s) => Math.max(m, s.length), 0);
  for (let rank = 0; rank < maxLen; rank++) {
    for (const sub of perProvider) if (rank < sub.length) out.push(sub[rank]);
  }
  return out;
}
