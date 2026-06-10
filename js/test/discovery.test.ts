import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { save } from "../src/cache.js";
import { discover, toSpecs } from "../src/discovery.js";
import { OpenRouter } from "../src/providers/openrouter.js";

describe("toSpecs", () => {
  it("detects reasoning by name and orders it last", () => {
    const specs = toSpecs(
      [
        { id: "vendor/gpt-oss-120b", context_length: 8192 },
        { id: "vendor/llama-3.3-70b", context_length: 8192 },
      ],
      false,
    );
    expect(specs[0].id).toBe("vendor/llama-3.3-70b");
    expect(specs[specs.length - 1].id).toBe("vendor/gpt-oss-120b");
    expect(specs[specs.length - 1].tags).toContain("reasoning");
  });

  it("filters non-chat models", () => {
    const ids = toSpecs(
      [
        { id: "vendor/chat-70b", context_length: 8192 },
        { id: "whisper-large-v3" },
        { id: "vendor/text-embedding-3" },
        { id: "playai-tts" },
        { id: "vendor/llama-guard-8b", context_length: 8192 },
      ],
      false,
    ).map((s) => s.id);
    expect(ids).toEqual(["vendor/chat-70b"]);
  });

  it("free_only filter + order (large, small, reasoning)", () => {
    const ids = toSpecs(
      [
        { id: "vendor/big-70b:free", context_length: 131072, supported_parameters: ["tools"] },
        { id: "vendor/small-8b:free", context_length: 8192 },
        { id: "vendor/think-70b:free", context_length: 200000, supported_parameters: ["tools", "reasoning"] },
        { id: "vendor/paid-70b", context_length: 1000 },
      ],
      true,
    ).map((s) => s.id);
    expect(ids).toEqual(["vendor/big-70b:free", "vendor/small-8b:free", "vendor/think-70b:free"]);
  });
});

describe("discover", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    delete process.env.FREELM_CACHE_DIR;
  });

  it("refetches live when the cached list yields no usable specs", async () => {
    process.env.FREELM_CACHE_DIR = mkdtempSync(join(tmpdir(), "freelm-test-"));
    save("openrouter", [{ id: "whisper-large-v3" }]); // filters down to nothing
    const live = { data: [{ id: "vendor/big-70b:free", context_length: 131072 }] };
    const fetchMock = vi.fn(async () => new Response(JSON.stringify(live), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const p = new OpenRouter("k");
    expect(await discover(p)).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1); // stale-empty cache did not block the live fetch
    expect(p.models.map((m) => m.id)).toEqual(["vendor/big-70b:free"]);
  });
});
