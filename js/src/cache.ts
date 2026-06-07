/** Tiny TTL disk cache for discovered model lists. Mirrors the Python impl. */
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { wallS } from "./time.js";

const DEFAULT_TTL = 3600;

export function cacheDir(): string {
  return process.env.FREELM_CACHE_DIR || path.join(os.homedir(), ".cache", "freelm");
}

export function defaultTtl(): number {
  const r = Number(process.env.FREELM_CACHE_TTL);
  return Number.isFinite(r) && r > 0 ? r : DEFAULT_TTL;
}

function cachePath(name: string): string {
  return path.join(cacheDir(), `models-${name.replace(/\//g, "_")}.json`);
}

export function load(name: string): any[] | null {
  try {
    const entry = JSON.parse(fs.readFileSync(cachePath(name), "utf-8"));
    if (wallS() > (entry.expires_at ?? 0)) return null;
    return entry.data ?? null;
  } catch {
    return null;
  }
}

export function save(name: string, data: any[], ttl?: number | null): void {
  try {
    fs.mkdirSync(cacheDir(), { recursive: true });
    const entry = { data, expires_at: wallS() + (ttl ?? defaultTtl()) };
    fs.writeFileSync(cachePath(name), JSON.stringify(entry), { mode: 0o600 });
  } catch {
    // best-effort cache; never fatal
  }
}

export function clear(name: string): void {
  try {
    fs.rmSync(cachePath(name));
  } catch {
    // ignore
  }
}
