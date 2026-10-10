/** Candidate ordering strategies. A candidate is one (provider, key, model). */
import { ConfigError } from "./errors.js";
import type { KeyState } from "./keys.js";
import { isVirtual } from "./registry.js";

export const STRATEGIES = ["smart", "priority", "round_robin", "quota_aware", "latency"] as const;
export type Strategy = (typeof STRATEGIES)[number];

export interface Candidate {
  provider: any;
  key: KeyState;
  model: string;
}

/** [provider, modelIds] pairs this alias may use, in the given order. A
 * concrete id goes only to providers that list it (an id nobody lists passes
 * through to all); a provider whose free guard rejects it is skipped and its
 * ConfigError returned, so the caller can throw it when nothing else serves. */
export function routable(providers: any[], alias: string | string[]): { routes: Array<[any, string[]]>; guard: ConfigError | null } {
  const aliases = Array.isArray(alias) ? alias : [alias];
  const owners = new Map<string, Set<any>>();
  for (const a of aliases) if (!isVirtual(a)) owners.set(a, new Set(providers.filter((p) => p.knowsModel(a))));
  const routes: Array<[any, string[]]> = [];
  let guard: ConfigError | null = null;
  for (const p of providers) {
    const chain = aliases.filter((a) => !owners.get(a)?.size || owners.get(a)!.has(p));
    if (!chain.length) continue;
    try {
      const models = p.resolveModels(chain);
      if (models.length) routes.push([p, models]);
    } catch (e) {
      if (!(e instanceof ConfigError)) throw e;
      guard ??= e;
    }
  }
  return { routes, guard };
}

export function orderCandidates(
  providers: any[],
  alias: string | string[],
  now: number,
  strategy: string,
  rr: { p: number },
): Candidate[] {
  let provs = [...providers];

  // provider `priority` is the universal tiebreak: primary for priority,
  // secondary for the dynamic strategies, baseline order for round_robin.
  if (strategy === "round_robin" && provs.length) {
    provs.sort((a, b) => a.priority - b.priority);
    const i = (rr.p ?? 0) % provs.length;
    provs = [...provs.slice(i), ...provs.slice(0, i)];
    rr.p = (rr.p ?? 0) + 1;
  } else if (strategy === "quota_aware") {
    provs.sort((a, b) => b.capacity(now) - a.capacity(now) || a.priority - b.priority);
  } else if (strategy === "latency") {
    // Infinity - Infinity is NaN (falsy) -> the priority tiebreak kicks in
    provs.sort((a, b) => a.avgLatency() - b.avgLatency() || a.priority - b.priority);
  } else if (strategy === "smart") {
    // priority tiers first; within a tier the fastest measured provider — an
    // unknown one counts as typical, so it gets tried and measured
    provs.sort((a, b) => a.priority - b.priority || a.expectedLatency(now) - b.expectedLatency(now));
  } else {
    provs.sort((a, b) => a.priority - b.priority);
  }

  const { routes, guard } = routable(provs, alias);

  // Build each provider's own ordered sublist (rotated keys, then models).
  // NB: benched models keep their rank slot here and are skipped at selection
  // time — dropping them would promote the provider's next (often equally
  // dead) model to rank 0 and starve the interleave.
  const perProvider: Candidate[][] = [];
  for (const [p, models] of routes) {
    let keys = [...p.keys];
    if (keys.length) {
      const ki = p._rr % keys.length;
      keys = [...keys.slice(ki), ...keys.slice(0, ki)];
      p._rr++;
    }
    const sub: Candidate[] = [];
    for (const k of keys) for (const mid of models) sub.push({ provider: p, key: k, model: mid });
    if (sub.length) perProvider.push(sub);
  }
  if (!perProvider.length && guard) throw guard;

  // Interleave breadth-first across providers (best model of each, then next).
  const out: Candidate[] = [];
  const maxLen = perProvider.reduce((m, s) => Math.max(m, s.length), 0);
  for (let rank = 0; rank < maxLen; rank++) {
    for (const sub of perProvider) if (rank < sub.length) out.push(sub[rank]);
  }
  return out;
}
