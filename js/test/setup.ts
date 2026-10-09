// Never read or write the developer's real ~/.cache/freelm from unit tests:
// a cached live model list would silently change what the router picks.
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeEach } from "vitest";

beforeEach(() => {
  process.env.FREELM_CACHE_DIR = mkdtempSync(join(tmpdir(), "freelm-test-cache-"));
  delete process.env.FREELM_PERSIST;
});
