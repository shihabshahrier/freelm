/** Opt-in persistent key state (rpd counters, cooldowns, disabled flags).
 *
 * Survives process restarts so a fresh run doesn't re-burn keys that are
 * already exhausted or dead. JSON file in the cache dir, 0600, atomic replace.
 * The schema is shared with the Python package (`src/freelm/_state.py`):
 *
 *   {"<provider>:<sha256(key)[:12]>": {rpd_used, rpd_reset_wall,
 *    cooldown_until_wall, disabled, last_error}}
 *
 * Raw keys are never written — only a short hash. Wall-clock timestamps in the
 * file are converted to/from the in-process monotonic clock on load/save.
 * Multi-process use is last-writer-wins (best effort, documented).
 */
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import { cacheDir } from "./cache.js";
import { wallS } from "./time.js";

function keyId(providerName: string, key: string): string {
  return `${providerName}:${createHash("sha256").update(key, "utf-8").digest("hex").slice(0, 12)}`;
}

export class StateStore {
  path: string;

  constructor(filePath?: string) {
    this.path = filePath ?? path.join(cacheDir(), "state.json");
  }

  private read(): Record<string, any> {
    try {
      const data = JSON.parse(fs.readFileSync(this.path, "utf-8"));
      return data && typeof data === "object" && !Array.isArray(data) ? data : {};
    } catch {
      return {};
    }
  }

  loadInto(providers: any[], nowMono: number): void {
    const data = this.read();
    if (!Object.keys(data).length) return;
    const wall = wallS();
    for (const p of providers) {
      for (const k of p.keys) {
        const e = data[keyId(p.name, k.key)];
        if (!e || typeof e !== "object") continue;
        k.rpdUsed = Math.trunc(Number(e.rpd_used) || 0);
        const rr = Number(e.rpd_reset_wall) || 0;
        if (rr > wall) k.rpdReset = nowMono + (rr - wall);
        const cu = Number(e.cooldown_until_wall) || 0;
        if (cu > wall) k.cooldownUntil = nowMono + (cu - wall);
        k.disabled = Boolean(e.disabled);
        if (e.last_error) k.lastError = String(e.last_error);
      }
    }
  }

  save(providers: any[], nowMono: number): void {
    // merge over existing entries so other processes/providers aren't clobbered
    const data = this.read();
    const wall = wallS();
    for (const p of providers) {
      for (const k of p.keys) {
        data[keyId(p.name, k.key)] = {
          rpd_used: k.rpdUsed,
          rpd_reset_wall: k.rpdReset > 0 ? wall + (k.rpdReset - nowMono) : 0,
          cooldown_until_wall: k.cooldownUntil > nowMono ? wall + (k.cooldownUntil - nowMono) : 0,
          disabled: k.disabled,
          last_error: k.lastError,
        };
      }
    }
    const tmp = this.path + ".tmp";
    try {
      fs.mkdirSync(path.dirname(this.path), { recursive: true });
      fs.writeFileSync(tmp, JSON.stringify(data), { mode: 0o600 });
      fs.renameSync(tmp, this.path);
    } catch {
      // persistence is best-effort; never fatal
    }
  }
}
