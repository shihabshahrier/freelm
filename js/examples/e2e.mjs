// Live smoke test for the JS port. Reads keys from env only.
//   set -a; . ../.env; set +a
//   node examples/e2e.mjs
import { FreeLLM, providersFromEnv, listFreeModels } from "../dist/index.js";

const provs = providersFromEnv();
console.log("freelm-js | providers:", provs.map((p) => p.name).join(", "));

try {
  const specs = await listFreeModels(undefined, true);
  console.log(`discovery: ${specs.length} OpenRouter free models; top=${specs[0]?.id}`);
} catch (e) {
  console.log("discovery failed:", e?.message ?? e);
}

const llm = new FreeLLM(providersFromEnv(), { strategy: "quota_aware", timeout: 25 });

try {
  const r = await llm.chat("Reply with exactly one word: pong", { max_tokens: 10, temperature: 0 });
  console.log(`chat   -> ${r.provider}/${r.model} ${r.latencyMs.toFixed(0)}ms: ${JSON.stringify(r.text)}`);
} catch (e) {
  console.log("chat FAIL:", e?.constructor?.name, String(e?.message ?? e).slice(0, 100));
}

try {
  process.stdout.write("stream -> ");
  for await (const c of llm.stream("Count: 1 2 3", { max_tokens: 20, temperature: 0 })) process.stdout.write(c);
  process.stdout.write("\n");
} catch (e) {
  console.log("stream FAIL:", e?.constructor?.name, String(e?.message ?? e).slice(0, 100));
}

console.log("health:");
for (const row of llm.health()) console.log(`   ${row.provider.padEnd(11)} ready=${row.ready} last_error=${row.lastError}`);
console.log("done.");
