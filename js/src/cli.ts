/** freelm CLI — chat / models / health from the terminal (npx freelm ...).
 *
 *   freelm chat "explain failover in one line" [--model auto] [--stream] [--strategy priority]
 *   freelm models [--provider openrouter]
 *   freelm health
 *   freelm --version
 *
 * Keys come from the environment (same vars as the library). Zero deps.
 */
import { FreeLLM } from "./client.js";
import { providersFromEnv } from "./config.js";
import { discover } from "./discovery.js";
import { ConfigError, FreeLLMError } from "./errors.js";
import { STRATEGIES, Strategy } from "./strategy.js";
import { VERSION } from "./version.js";

const HELP = `freelm ${VERSION} — free, always-up LLM client over free-tier providers.

usage:
  freelm chat <prompt> [--model <alias|id>] [--strategy <s>] [--stream]
  freelm models [--provider <name>]
  freelm health
  freelm --version

strategies: ${STRATEGIES.join(" | ")}
`;

function flag(args: string[], name: string): string | undefined {
  const i = args.indexOf(name);
  if (i < 0) return undefined;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
}

function boolFlag(args: string[], name: string): boolean {
  const i = args.indexOf(name);
  if (i < 0) return false;
  args.splice(i, 1);
  return true;
}

async function cmdChat(args: string[]): Promise<number> {
  const model = flag(args, "--model") ?? "auto";
  const strategy = (flag(args, "--strategy") ?? "priority") as Strategy;
  const stream = boolFlag(args, "--stream");
  const prompt = args.join(" ").trim();
  if (!prompt) {
    process.stderr.write("usage: freelm chat <prompt> [--model X] [--stream]\n");
    return 2;
  }
  const llm = new FreeLLM(providersFromEnv(), { strategy });
  if (stream) {
    for await (const chunk of llm.stream(prompt, { model })) process.stdout.write(chunk);
    process.stdout.write("\n");
  } else {
    const r = await llm.chat(prompt, { model });
    process.stdout.write(r.text + "\n");
    process.stderr.write(`[${r.provider}/${r.model}]\n`);
  }
  return 0;
}

async function cmdModels(args: string[]): Promise<number> {
  const only = flag(args, "--provider");
  let provs = providersFromEnv();
  if (only) {
    provs = provs.filter((p) => p.name === only);
    if (!provs.length) {
      process.stderr.write(`no provider named '${only}' configured\n`);
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

function cmdHealth(): number {
  const llm = new FreeLLM(providersFromEnv());
  for (const row of llm.health()) {
    process.stdout.write(
      `${String(row.provider).padEnd(11)} ${String(row.key).padEnd(18)} ready=${String(row.ready).padEnd(5)} ` +
        `breaker=${String(row.breaker).padEnd(9)} rpd=${row.rpdUsed}/${row.rpd ?? "-"} lastError=${row.lastError}\n`,
    );
  }
  return 0;
}

export async function main(argv: string[]): Promise<number> {
  const args = [...argv];
  if (!args.length || args.includes("--help") || args.includes("-h")) {
    process.stdout.write(HELP);
    return 0;
  }
  if (args.includes("--version")) {
    process.stdout.write(`freelm ${VERSION}\n`);
    return 0;
  }
  const cmd = args.shift();
  try {
    if (cmd === "chat") return await cmdChat(args);
    if (cmd === "models") return await cmdModels(args);
    if (cmd === "health") return cmdHealth();
    process.stderr.write(`unknown command '${cmd}'\n\n${HELP}`);
    return 2;
  } catch (e) {
    if (e instanceof ConfigError) {
      process.stderr.write(`config error: ${e.message}\n`);
      return 2;
    }
    if (e instanceof FreeLLMError) {
      process.stderr.write(`error: ${e.message}\n`);
      return 1;
    }
    throw e;
  }
}
