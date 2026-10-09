/** Opt-in persistent key state (rpd counters, cooldowns, disabled flags).
 *
 * Survives process restarts so a fresh run doesn't re-burn keys that are
 * already exhausted or dead. JSON file in the cache dir, 0600, atomic replace.
 * The schema is shared with the Python package (`src/freelm/_state.py`):
 *
 *   {"<provider>:<sha256(key)[:12]>": {rpd_used, rpd_reset_wall,
 *    cooldown_until_wall, disabled, disabled_since_wall, last_error}}
 *
 * Raw keys are never written — only a short hash. Wall-clock timestamps in the
 * file are converted to/from the in-process monotonic clock on load/save. A
 * persisted `disabled` flag expires after DISABLED_TTL, so a dead key gets one
 * fresh try per day. Multi-process use is last-writer-wins (best effort).
 * A no-op where the runtime has no filesystem/crypto built-ins.
 */
import { cacheDir } from "./cache.js";
import { builtin, joinPath } from "./runtime.js";
import { wallS } from "./time.js";

export const DISABLED_TTL = 86400;

export function keyId(providerName: string, key: string): string | null {
  const crypto = builtin("node:crypto");
  if (!crypto) return null;
  return `${providerName}:${crypto.createHash("sha256").update(key, "utf-8").digest("hex").slice(0, 12)}`;
}

function num(v: any): number {
  const n = Number(v ?? 0);
  return Number.isFinite(n) ? n : 0;
}

export class StateStore {
  path: string | null;

  constructor(filePath?: string) {
    const dir = cacheDir();
    this.path = filePath ?? (dir ? joinPath(dir, "state.json") : null);
  }

  private read(): Record<string, any> {
    const fs = builtin("node:fs");
    if (!fs || !this.path) return {};
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
        const id = keyId(p.name, k.key);
        const e = id ? data[id] : null;
        if (!e || typeof e !== "object") continue;
        const rr = num(e.rpd_reset_wall);
        if (rr > wall) {
          // same daily window as when saved -> keep its count (else a new day)
          k.rpdUsed = Math.trunc(num(e.rpd_used));
          k.rpdReset = nowMono + (rr - wall);
        }
        const cu = num(e.cooldown_until_wall);
        if (cu > wall) k.cooldownUntil = nowMono + (cu - wall);
        const since = num(e.disabled_since_wall) || wall;
        if (Boolean(e.disabled) && wall - since < DISABLED_TTL) {
          k.disabled = true;
          k.disabledSinceWall = since;
        } else {
          k.disabledSinceWall = 0; // expired: if it fails again, that's a fresh disable
        }
        if (typeof e.last_error === "string") k.lastError = e.last_error;
      }
    }
  }

  save(providers: any[], nowMono: number): void {
    const fs = builtin("node:fs");
    if (!fs || !this.path) return;
    // merge over existing entries so other processes/providers aren't clobbered
    const data = this.read();
    const wall = wallS();
    for (const p of providers) {
      for (const k of p.keys) {
        const id = keyId(p.name, k.key);
        if (!id) return;
        let since = 0;
        if (k.disabled) {
          const prev = data[id] && typeof data[id] === "object" ? data[id] : {};
          let inherited = prev.disabled ? num(prev.disabled_since_wall) : 0;
          if (inherited && wall - inherited >= DISABLED_TTL) inherited = 0; // another process' expired disable
          since = k.disabledSinceWall || inherited || wall;
          k.disabledSinceWall = since;
        }
        data[id] = {
          rpd_used: k.rpdUsed,
          rpd_reset_wall: k.rpdReset > 0 ? wall + (k.rpdReset - nowMono) : 0,
          cooldown_until_wall: k.cooldownUntil > nowMono ? wall + (k.cooldownUntil - nowMono) : 0,
          disabled: k.disabled,
          disabled_since_wall: since,
          last_error: k.lastError,
        };
      }
    }
    // unique temp file per writer: concurrent processes must never swap each
    // other's half-written file into place
    const tmp = `${this.path}.${(globalThis as any).process?.pid ?? 0}.${Math.random().toString(36).slice(2)}.tmp`;
    try {
      const dir = this.path.slice(0, Math.max(this.path.lastIndexOf("/"), this.path.lastIndexOf("\\")));
      if (dir) fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(tmp, JSON.stringify(data), { mode: 0o600 });
      fs.renameSync(tmp, this.path);
    } catch {
      try {
        fs.rmSync(tmp, { force: true });
      } catch {
        /* ignore */
      }
      // persistence is best-effort; never fatal
    }
  }
}
