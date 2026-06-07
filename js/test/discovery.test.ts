import { describe, expect, it } from "vitest";
import { toSpecs } from "../src/discovery.js";

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
