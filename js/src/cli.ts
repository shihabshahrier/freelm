/** freelm CLI — chat / models / health / doctor / serve (npx freelm ...).
 *
 *   freelm chat "explain failover in one line" [--model auto] [--stream] [--strategy priority]
 *   freelm models [--provider openrouter]
 *   freelm health
 *   freelm doctor [--json]                    # live-check every key, with fixes
 *   freelm serve [--port 4000] [--api-key K]  # local OpenAI-compatible endpoint
 *   freelm --version
 *
 * Keys come from the environment (same vars as the library). Zero deps.
 */
import { parseArgs } from "node:util";
import { FreeLLM } from "./client.js";
import { KEYLESS, PROVIDER_ENV, envKeys, keylessMode, providersFromEnv } from "./config.js";
import { discover } from "./discovery.js";
import {
  AuthError,
  BadRequest,
  ConfigError,
  FreeLLMError,
  ModelNotFound,
  NoProvidersAvailable,
  QuotaExhausted,
  RateLimited,
  Transient,
} from "./errors.js";
import { env } from "./runtime.js";
import { STRATEGIES, Strategy } from "./strategy.js";
import { VERSION } from "./version.js";

const HELP = `freelm ${VERSION} — free, always-up LLM client over free-tier providers.

usage:
  freelm chat <prompt> [--model <alias|id>] [--strategy <s>] [--stream]
  freelm models [--provider <name>]
  freelm health                 per-key state of this process (no network)
  freelm doctor [--json]        live-check every configured key, with fixes
  freelm serve [--host 127.0.0.1] [--port 4000] [--api-key <k>] [--cors] [--no-fallback] [--quiet]
                                local OpenAI-compatible endpoint (/v1/chat/completions, /v1/models)
  freelm --version

strategies: ${STRATEGIES.join(" | ")}
`;

class UsageError extends Error {}

/** The CLI is for trying things: with no keys it falls back to the keyless
 * endpoints (with a notice) unless FREELM_KEYLESS says otherwise. */
const keyless = () => env("FREELM_KEYLESS") || "auto";

function strategyOf(v: unknown): Strategy {
  const s = String(v ?? "priority");
  if (!STRATEGIES.includes(s as Strategy)) throw new UsageError(`unknown strategy '${s}' (pick one of ${STRATEGIES.join(", ")})`);
  return s as Strategy;
}

async function cmdChat(argv: string[]): Promise<number> {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: { model: { type: "string", default: "auto" }, strategy: { type: "string" }, stream: { type: "boolean", default: false } },
  });
  const prompt = positionals.join(" ").trim();
  if (!prompt) throw new UsageError("usage: freelm chat <prompt> [--model X] [--stream]");
  const llm = new FreeLLM(providersFromEnv(keyless()), { strategy: strategyOf(values.strategy) });
  if (values.stream) {
    for await (const chunk of llm.stream(prompt, { model: values.model })) process.stdout.write(chunk);
    process.stdout.write("\n");
  } else {
    const r = await llm.chat(prompt, { model: values.model });
    process.stdout.write(r.text + "\n");
    process.stderr.write(`[${r.provider}/${r.model}]\n`);
  }
  return 0;
}

async function cmdModels(argv: string[]): Promise<number> {
  const { values } = parseArgs({ args: argv, options: { provider: { type: "string" } } });
  let provs = providersFromEnv(keyless());
  if (values.provider) {
    provs = provs.filter((p) => p.name === values.provider);
    if (!provs.length) {
      process.stderr.write(`no provider named '${values.provider}' configured\n`);
      return 2;
    }
  }
  for (const p of provs) {
    if (p.discover && !p._discovered) await discover(p);
    process.stdout.write(`${p.name}:\n`);
    for (const m of p.models) {
      const ctx = m.ctx ? ` ctx=${m.ctx}` : "";
      process.stdout.write(`  ${m.id}  [${m.tags.join(",")}]${ctx}\n`);
    }
  }
  return 0;
}

function cmdHealth(argv: string[]): number {
  parseArgs({ args: argv, options: {} });
  const llm = new FreeLLM(providersFromEnv(keyless()));
  for (const row of llm.health()) {
    process.stdout.write(
      `${String(row.provider).padEnd(11)} ${String(row.key).padEnd(18)} ready=${String(row.ready).padEnd(5)} ` +
        `breaker=${String(row.breaker).padEnd(9)} rpd=${row.rpdUsed}/${row.rpd ?? "-"} lastError=${row.lastError}\n`,
    );
  }
  process.stderr.write("(state of this process only — run `freelm doctor` to test the keys live)\n");
  return 0;
}

// -- doctor ----------------------------------------------------------------------

/** The human part of a provider error body (JSON message/detail). */
function short(e: any): string {
  let msg: string = e?.detail ?? e?.message ?? String(e);
  try {
    let obj = JSON.parse(msg);
    if (Array.isArray(obj) && obj.length) obj = obj[0];
    if (obj && typeof obj === "object") {
      const err = obj.error ?? obj;
      if (err && typeof err === "object") msg = err.message ?? err.detail ?? err.title ?? msg;
      else if (typeof err === "string") msg = err;
    }
  } catch {
    /* not JSON */
  }
  return String(msg).split(/\s+/).join(" ").slice(0, 110);
}

function diagnose(e: any, signup: string): [string, string, boolean] {
  if (!e) return ["FAIL", "no model could be tried (all retired or benched)", false];
  const code = e.status || "";
  const s = short(e);
  if (e instanceof AuthError) return ["FAIL", `key rejected (${code}): ${s} — get a new free key: ${signup}`, false];
  if (e instanceof QuotaExhausted) return ["FAIL", `no free quota/credits left (${code}): ${s}`, false];
  if (e instanceof RateLimited) return ["WARN", `key works but is rate-limited right now (${code})`, true];
  if (e instanceof ModelNotFound) return ["FAIL", `no usable model (${code}): ${s}`, false];
  if (e instanceof Transient) return ["WARN", `provider unreachable or overloaded (${code || "network"}): ${s}`, false];
  if (e instanceof BadRequest) return ["FAIL", `request rejected (${code}): ${s}`, false];
  return ["FAIL", s, false];
}

/** One tiny real request through `p` (a one-provider client). */
async function check(p: any, signup: string, timeout: number): Promise<Record<string, any>> {
  const row: Record<string, any> = { provider: p.name, key: p.keys[0].masked(), model: null, latency_ms: null };
  const llm = new FreeLLM([p], { maxAttempts: 4, timeout, persist: false }); // test the key, not saved state
  try {
    const r = await llm.chat("Reply with the single word: ok", { max_tokens: 16, temperature: 0 });
    Object.assign(row, { status: "OK", works: true, model: r.model, latency_ms: Math.round(r.latencyMs), detail: `${r.model} · ${Math.round(r.latencyMs)} ms` });
  } catch (e: any) {
    if (!(e instanceof FreeLLMError)) throw e;
    const last = e instanceof NoProvidersAvailable ? e.attempts[e.attempts.length - 1]?.[1] : e;
    const [status, detail, works] = diagnose(last, signup);
    Object.assign(row, { status, works, detail });
  }
  return row;
}

async function cmdDoctor(argv: string[]): Promise<number> {
  const { values } = parseArgs({ args: argv, options: { json: { type: "boolean", default: false }, timeout: { type: "string", default: "30" } } });
  const timeout = Number(values.timeout) || 30;
  const configured = PROVIDER_ENV.map((spec) => [spec, envKeys(spec)] as const);
  const missing = configured.filter(([, k]) => !k.length).map(([s]) => s);
  const haveKeys = configured.some(([, k]) => k.length);
  const checkKeyless = keylessMode(keyless()) !== "never";
  const missingJson = missing.map((s) => ({ provider: s.name, env: s.keyVars[0], signup: s.signupUrl }));
  const say = (line: string) => {
    if (!values.json) process.stdout.write(line + "\n");
  };

  if (haveKeys) say(`freelm ${VERSION} doctor — ${configured.reduce((a, [, k]) => a + k.length, 0)} key(s), one tiny request each\n`);
  else {
    say("no provider keys found in the environment. Free keys (no credit card needed for most):\n");
    for (const s of missing) say(`  ${s.name.padEnd(11)} export ${s.keyVars[0]}=...   ${s.signupUrl}`);
    say("\nSet one or more, then run `freelm doctor` again.");
  }
  const line = (r: Record<string, any>) => `  ${r.provider.padEnd(11)} ${String(r.key).padEnd(16)} ${r.status.padEnd(4)}  ${r.detail}`;
  const rows: Record<string, any>[] = [];
  for (const [spec, keys] of configured) {
    for (const key of keys) {
      const row = await check(new spec.cls([key], { tier: env(spec.tierVar) ?? "free" }), spec.signupUrl, timeout);
      rows.push(row);
      say(line(row));
    }
  }
  const keylessRows: Record<string, any>[] = [];
  if (checkKeyless) {
    say("\nkeyless endpoints (no signup; low limits; free routes may log prompts):");
    const keyed = new Set(configured.filter(([, k]) => k.length).map(([s]) => s.name));
    for (const cls of KEYLESS) {
      const p = new (cls as any)();
      if (keyed.has(p.name)) continue; // already checked with your key
      const row = await check(p, "", timeout);
      keylessRows.push(row);
      say(line(row));
    }
  }
  const working = rows.filter((r) => r.works);
  const keylessOk = keylessRows.filter((r) => r.works);
  if (values.json) process.stdout.write(JSON.stringify({ keys: rows, keyless: keylessRows, missing: missingJson }, null, 2) + "\n");
  else {
    if (haveKeys && missing.length) {
      say("\nnot configured (more free capacity, more failover):");
      for (const s of missing) say(`  ${s.name.padEnd(11)} ${s.keyVars[0].padEnd(22)} ${s.signupUrl}`);
    }
    const ready = [...new Set([...working, ...keylessOk].map((r) => r.provider))].sort();
    const parts = [rows.length ? `${working.length} of ${rows.length} key(s) working` : "no keys configured"];
    if (keylessRows.length) parts.push(`${keylessOk.length} keyless endpoint(s) up`);
    say(`\n${parts.join(", ")}${ready.length ? ` — ready: ${ready.join(", ")}` : ""}`);
  }
  if (working.length || keylessOk.length) return 0;
  return haveKeys ? 1 : 2;
}

async function cmdServe(argv: string[]): Promise<number> {
  const { values } = parseArgs({
    args: argv,
    options: {
      host: { type: "string", default: env("FREELM_HOST") ?? "127.0.0.1" },
      port: { type: "string", default: env("FREELM_PORT") ?? "4000" },
      "api-key": { type: "string", default: env("FREELM_SERVER_KEY") },
      strategy: { type: "string" },
      cors: { type: "boolean", default: false },
      "no-fallback": { type: "boolean", default: false },
      quiet: { type: "boolean", default: false },
    },
  });
  const port = Number(values.port);
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new UsageError(`invalid --port '${values.port}'`);
  const { serve } = await import("./server.js");
  const server = await serve({
    providers: providersFromEnv(keyless()),
    host: values.host,
    port,
    apiKey: values["api-key"] ?? null,
    cors: values.cors,
    fallbackModel: values["no-fallback"] ? null : "auto",
    quiet: values.quiet,
    strategy: strategyOf(values.strategy),
  });
  await new Promise<void>((resolve) => {
    const stop = () => server.close(() => resolve());
    process.once("SIGINT", stop);
    process.once("SIGTERM", stop);
  });
  return 0;
}

const COMMANDS: Record<string, (argv: string[]) => Promise<number> | number> = {
  chat: cmdChat,
  models: cmdModels,
  health: cmdHealth,
  doctor: cmdDoctor,
  serve: cmdServe,
};

export async function main(argv: string[]): Promise<number> {
  const [cmd, ...rest] = argv;
  if (!cmd || cmd === "--help" || cmd === "-h" || cmd === "help") {
    process.stdout.write(HELP);
    return 0;
  }
  if (cmd === "--version" || cmd === "-V") {
    process.stdout.write(`freelm ${VERSION}\n`);
    return 0;
  }
  const run = COMMANDS[cmd];
  if (!run) {
    process.stderr.write(`unknown command '${cmd}'\n\n${HELP}`);
    return 2;
  }
  if (rest.includes("--help") || rest.includes("-h")) {
    process.stdout.write(HELP);
    return 0;
  }
  try {
    return await run(rest);
  } catch (e: any) {
    if (e instanceof UsageError || e?.code?.startsWith?.("ERR_PARSE_ARGS")) {
      process.stderr.write(`usage error: ${e.message}\n`);
      return 2;
    }
    if (e instanceof ConfigError) {
      process.stderr.write(`config error: ${e.message}\n`);
      return 2;
    }
    if (e instanceof FreeLLMError) {
      process.stderr.write(`error: ${e.message}\n`);
      return 1;
    }
    if (e?.code === "EADDRINUSE" || e?.code === "EACCES") {
      process.stderr.write(`error: ${e.message}\n`);
      return 1;
    }
    throw e;
  }
}
