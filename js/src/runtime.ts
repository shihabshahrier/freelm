/** Runtime access without static `node:` imports, so the library bundles and
 * runs on Node, Bun, Deno, Cloudflare Workers, edge functions and browsers.
 * Disk cache and persistence switch on only where Node built-ins exist
 * (`process.getBuiltinModule`, Node >= 20.16 / 22.3, Bun); elsewhere they are
 * silently skipped — both are best-effort by design. (TS-only: Python always
 * has the stdlib.) */

/** An environment variable, or undefined where there is no `process.env`. */
export function env(name: string): string | undefined {
  try {
    const v = (globalThis as any).process?.env?.[name];
    return typeof v === "string" ? v : undefined;
  } catch {
    return undefined;
  }
}

/** A Node built-in module (e.g. "node:fs"), or null when unavailable. */
export function builtin<T = any>(name: string): T | null {
  try {
    const get = (globalThis as any).process?.getBuiltinModule;
    return typeof get === "function" ? ((get(name) as T) ?? null) : null;
  } catch {
    return null;
  }
}

/** Join path segments with the platform separator (forward slashes work on Windows too). */
export function joinPath(...parts: string[]): string {
  const path = builtin<{ join: (...p: string[]) => string }>("node:path");
  return path ? path.join(...parts) : parts.join("/").replace(/\/{2,}/g, "/");
}

export function homeDir(): string | null {
  const os = builtin<{ homedir: () => string }>("node:os");
  return os?.homedir() ?? env("HOME") ?? env("USERPROFILE") ?? null;
}

/** Don't let a pending timer keep the process alive (no-op outside Node/Bun). */
export function unref(t: any): void {
  try {
    t?.unref?.();
  } catch {
    /* ignore */
  }
}
