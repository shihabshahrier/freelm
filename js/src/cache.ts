/** Tiny TTL disk cache for discovered model lists. Mirrors the Python impl.
 * Path: $FREELM_CACHE_DIR or ~/.cache/freelm/models-<provider>.json (0600).
 * Best-effort: a no-op where the runtime has no filesystem. */
import { builtin, env, homeDir, joinPath } from "./runtime.js";
import { wallS } from "./time.js";

const DEFAULT_TTL = 3600;

export function cacheDir(): string | null {
  const d = env("FREELM_CACHE_DIR");
  if (d) return d;
  const home = homeDir();
  return home ? joinPath(home, ".cache", "freelm") : null;
}

export function defaultTtl(): number {
  const raw = env("FREELM_CACHE_TTL");
  if (raw === undefined || raw === "") return DEFAULT_TTL;
  const r = Number(raw);
  return Number.isFinite(r) && r >= 0 ? r : DEFAULT_TTL; // 0 = don't reuse the cache
}

function cachePath(name: string): string | null {
  const dir = cacheDir();
  return dir ? joinPath(dir, `models-${name.replace(/\//g, "_")}.json`) : null;
}

export function load(name: string): any[] | null {
  const fs = builtin("node:fs");
  const p = cachePath(name);
  if (!fs || !p) return null;
  let entry: any;
  try {
    entry = JSON.parse(fs.readFileSync(p, "utf-8"));
  } catch {
    return null;
  }
  // foreign/corrupt file: ignore it, a live fetch will rewrite it
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) return null;
  const exp = Number(entry.expires_at);
  if (!Number.isFinite(exp) || wallS() > exp) return null;
  if (!Array.isArray(entry.data)) return null;
  return entry.data.filter((m: any) => m && typeof m === "object" && !Array.isArray(m));
}

export function save(name: string, data: any[], ttl?: number | null): void {
  const fs = builtin("node:fs");
  const dir = cacheDir();
  const p = cachePath(name);
  if (!fs || !dir || !p) return;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const entry = { data, expires_at: wallS() + (ttl ?? defaultTtl()) };
    fs.writeFileSync(p, JSON.stringify(entry), { mode: 0o600 });
    fs.chmodSync(p, 0o600); // mode above only applies when the file is created
  } catch {
    // best-effort cache; never fatal
  }
}

export function clear(name: string): void {
  const fs = builtin("node:fs");
  const p = cachePath(name);
  if (!fs || !p) return;
  try {
    fs.rmSync(p, { force: true });
  } catch {
    // ignore
  }
}
