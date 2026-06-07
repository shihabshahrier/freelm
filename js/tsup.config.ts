import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts", "src/compat/openai.ts"],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  sourcemap: true,
  target: "node18",
});
