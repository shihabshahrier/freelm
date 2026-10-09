import { defineConfig } from "tsup";

export default defineConfig({
  // ESM + CJS. `splitting` (experimental for CJS) shares one copy of the core
  // between `freelm`, `freelm/compat` and the CLI, so a FreeLLM or an error
  // class from one entry is the same class in the other.
  entry: { index: "src/index.ts", "compat/openai": "src/compat/openai.ts", cli: "src/cli.ts" },
  format: ["esm", "cjs"],
  dts: { entry: { index: "src/index.ts", "compat/openai": "src/compat/openai.ts" } },
  splitting: true,
  clean: true,
  sourcemap: false,
  target: "node20",
});
