// Reproducible provider-failure simulation for freelm (JavaScript / TypeScript).
// Same scenarios as benchmarks/failover.py: fake providers on localhost over
// real HTTP, freelm with its defaults (timeout 60 s, hedging on, smart routing),
// two calls per scenario on the same client.
//
//   cd js && npm run build && node ../benchmarks/failover.mjs            # local build
//   node benchmarks/failover.mjs --package freelm                         # an installed package
//   node benchmarks/failover.mjs --no-hedge                               # the pre-0.5 behaviour
import http from "node:http";
import { fileURLToPath, pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const pkgArg = args.includes("--package") ? args[args.indexOf("--package") + 1] : null;
const target = pkgArg ?? pathToFileURL(fileURLToPath(new URL("../js/dist/index.js", import.meta.url))).href;
const { FreeLLM, Provider, modelSpec, NoProvidersAvailable, VERSION } = await import(target);
const hedge = !args.includes("--no-hedge");

const OK = JSON.stringify({ id: "x", model: "m", choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }] });
const server = http.createServer((req, res) => {
  const kind = req.url.split("/")[1];
  req.resume();
  req.on("end", () => {
    if (kind === "hang") return; // accept, never answer
    if (kind === "stall" || kind === "sok") {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.flushHeaders();
      if (kind === "sok") setTimeout(() => res.end('data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'), 300);
      return;
    }
    const delay = { ok: 300, slow8: 8000 }[kind] ?? 0;
    const status = { r429: 429, e500: 500 }[kind] ?? 200;
    setTimeout(() => {
      res.writeHead(status, { "content-type": "application/json" });
      res.end(status === 200 ? OK : '{"error":{"message":"simulated"}}');
    }, delay);
  });
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const port = server.address().port;

const SCENARIOS = [
  ["429, then healthy", ["r429", "ok"], false],
  ["500, then healthy", ["e500", "ok"], false],
  ["connection refused, then healthy", ["dead", "ok"], false],
  ["hung provider, then healthy", ["hang", "ok"], false],
  ["unroutable host, then healthy", ["blackhole", "ok"], false],
  ["slow provider (8 s), then healthy", ["slow8", "ok"], false],
  ["stream stalls before first token, then healthy", ["stall", "sok"], true],
];
const url = (kind) =>
  kind === "dead" ? "http://127.0.0.1:9/v1" : kind === "blackhole" ? "http://10.255.255.1/v1" : `http://127.0.0.1:${port}/${kind}/v1`;
const providers = (kinds) =>
  kinds.map((k, i) => new Provider("k", { name: `${k}${i}`, baseUrl: url(k), models: [modelSpec("m", ["chat"])], rpm: null }));

console.log(`freelm ${VERSION} (Node ${process.versions.node}), hedge=${hedge ? "on" : "off"}, defaults otherwise (timeout 60 s)\n`);
console.log(`  ${"scenario".padEnd(48)} ${"1st call".padStart(16)}   ${"2nd call".padStart(16)}`);
for (const [label, kinds, stream] of SCENARIOS) {
  const llm = new FreeLLM(providers(kinds), { hedge, persist: false });
  const cells = [];
  for (let n = 0; n < 2; n++) {
    const t0 = performance.now();
    let ok = true;
    try {
      if (stream) for await (const _ of llm.stream("hi"));
      else await llm.chat("hi");
    } catch (e) {
      if (!(e instanceof NoProvidersAvailable)) throw e;
      ok = false;
    }
    const dt = (performance.now() - t0) / 1000;
    cells.push(`${dt.toFixed(1).padStart(5)} s${ok ? "" : " FAILED"}`);
  }
  console.log(`  ${label.padEnd(48)} ${cells[0].padStart(16)}   ${cells[1].padStart(16)}`);
}
server.closeAllConnections?.();
server.close();
process.exit(0);
