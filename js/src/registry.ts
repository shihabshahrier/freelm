/** Virtual-model registry: map aliases like `auto` / `chat:fast` to concrete ids. */

export interface ModelSpec {
  id: string;
  tags: string[];
  ctx: number;
  free: boolean;
}

export function modelSpec(id: string, tags: string[] = [], ctx = 0, free = true): ModelSpec {
  return { id, tags, ctx, free };
}

const SIZE_ALIASES: Record<string, string> = { best: "large", big: "large", mini: "small", cheap: "small", lite: "small" };
const VIRTUAL = new Set(["auto", "chat", "default", "large", "fast", "small", ...Object.keys(SIZE_ALIASES)]);

export function resolveModels(models: ModelSpec[], alias: string): string[] {
  const ids = models.map((m) => m.id);
  if (ids.includes(alias)) return [alias];

  const a = alias.trim().toLowerCase();
  const idx = a.indexOf(":");
  const base = idx >= 0 ? a.slice(0, idx) : a;
  const size = idx >= 0 ? a.slice(idx + 1) : "";

  if (!VIRTUAL.has(base)) return [alias]; // unknown -> a concrete model id (possibly with a suffix like ":free")

  let want = size || (["auto", "chat", "default"].includes(base) ? "" : base);
  want = SIZE_ALIASES[want] ?? want;

  if (want === "large" || want === "fast" || want === "small") {
    const sized = models.filter((m) => m.tags.includes(want)).map((m) => m.id);
    if (sized.length) return sized;
  }

  const chat = models.filter((m) => m.tags.includes("chat")).map((m) => m.id);
  return chat.length ? chat : ids;
}
