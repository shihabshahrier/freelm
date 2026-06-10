/** Virtual-model registry: map aliases like `auto` / `chat:fast` to concrete ids.
 * Give specs a `priority` to control order without replacing the list. */

export interface ModelSpec {
  id: string;
  tags: string[];
  ctx: number;
  free: boolean;
  priority: number; // lower = preferred (stable: ties keep list order)
}

export function modelSpec(id: string, tags: string[] = [], ctx = 0, free = true, priority = 0): ModelSpec {
  return { id, tags, ctx, free, priority };
}

const SIZE_ALIASES: Record<string, string> = { best: "large", big: "large", mini: "small", cheap: "small", lite: "small" };
// tags that can be asked for directly: `chat:tools`, `vision`, `reasoning`, ...
const TAG_ALIASES = new Set(["large", "fast", "small", "tools", "vision", "reasoning"]);
const VIRTUAL = new Set(["auto", "chat", "default", ...TAG_ALIASES, ...Object.keys(SIZE_ALIASES)]);

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

  const ordered = [...models].sort((x, y) => x.priority - y.priority); // stable: priority, then list order

  if (TAG_ALIASES.has(want)) {
    const tagged = ordered.filter((m) => m.tags.includes(want)).map((m) => m.id);
    if (tagged.length) return tagged;
  }

  const chat = ordered.filter((m) => m.tags.includes("chat")).map((m) => m.id);
  return chat.length ? chat : ordered.map((m) => m.id);
}
