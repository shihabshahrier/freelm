/** The user-facing async client: FreeLLM (chat, text, stream, health). */
import { discover } from "./discovery.js";
import * as engine from "./engine.js";
import { ConfigError, NoProvidersAvailable, ProviderError, RateLimited, Transient, classify } from "./errors.js";
import { Provider } from "./providers/base.js";
import { providersFromEnv } from "./config.js";
import { Candidate, STRATEGIES, Strategy } from "./strategy.js";
import { nowS, sleep } from "./time.js";
import { buildPayload, buildRequest, ChatRequest, ChatResponse, MessageLike } from "./types.js";

const UA = "freelm-js/0.1.0";

export interface FreeLLMOptions {
  strategy?: Strategy;
  maxAttempts?: number;
  timeout?: number; // seconds; also the overall per-call deadline
  wait?: boolean;
  maxWait?: number;
}

export type ChatOptions = { model?: string } & Record<string, any>;

function headersObj(res: Response): Record<string, string> {
  const o: Record<string, string> = {};
  res.headers.forEach((v, k) => (o[k.toLowerCase()] = v));
  return o;
}

function sseDelta(line: string): string | null {
  if (!line || !line.startsWith("data:")) return null;
  const data = line.slice(5).trim();
  if (!data || data === "[DONE]") return null;
  try {
    const obj = JSON.parse(data);
    return obj.choices?.[0]?.delta?.content || null;
  } catch {
    return null;
  }
}

export class FreeLLM {
  providers: Provider[];
  strategy: string;
  maxAttempts: number;
  timeout: number;
  wait: boolean;
  maxWait: number;
  private rr = { p: 0 };
  private discoveryDone = false;

  constructor(providers: Provider[], opts: FreeLLMOptions = {}) {
    if (!providers.length) throw new ConfigError("FreeLLM needs at least one provider");
    this.strategy = opts.strategy ?? "priority";
    if (!STRATEGIES.includes(this.strategy as Strategy)) {
      throw new ConfigError(`unknown strategy ${this.strategy}; pick one of ${STRATEGIES.join(", ")}`);
    }
    this.providers = providers;
    this.maxAttempts = opts.maxAttempts ?? 12;
    this.timeout = opts.timeout ?? 60;
    this.wait = opts.wait ?? false;
    this.maxWait = opts.maxWait ?? 20;
  }

  static fromEnv(opts: FreeLLMOptions = {}): FreeLLM {
    return new FreeLLM(providersFromEnv(), opts);
  }

  health(): Record<string, any>[] {
    const now = nowS();
    const out: Record<string, any>[] = [];
    for (const p of this.providers)
      for (const k of p.keys)
        out.push({
          provider: p.name,
          key: k.masked(),
          tier: k.tier,
          ready: k.ready(now),
          disabled: k.disabled,
          breaker: k.breaker.state,
          rpdUsed: k.rpdUsed,
          rpd: k.rpd,
          lastError: k.lastError,
          ewmaLatencyMs: Math.round(k.ewmaLatency * 10) / 10,
        });
    return out;
  }

  refreshModels(): void {
    this.discoveryDone = false;
    for (const p of this.providers) p._discovered = false;
  }

  private async ensureDiscovered(): Promise<void> {
    if (this.discoveryDone) return;
    await Promise.all(
      this.providers.map(async (p) => {
        if (p.discover) {
          try {
            await discover(p);
          } catch {
            /* keep fallback models */
          }
        }
      }),
    );
    this.discoveryDone = true;
  }

  private async fetchWithTimeout(url: string, init: RequestInit): Promise<Response> {
    if (!this.timeout) return fetch(url, init);
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), this.timeout * 1000);
    try {
      return await fetch(url, { ...init, signal: ac.signal });
    } finally {
      clearTimeout(timer);
    }
  }

  async chat(messages: MessageLike | MessageLike[], opts: ChatOptions = {}): Promise<ChatResponse> {
    await this.ensureDiscovered();
    const { model = "auto", ...params } = opts;
    const req = buildRequest(messages, model, params);
    const deadline = this.timeout ? nowS() + this.timeout : null;
    const attempts: Array<[Candidate, Error]> = [];
    let tried = new Set<string>();

    while (attempts.length < this.maxAttempts) {
      const now = nowS();
      if (deadline !== null && now >= deadline) break;
      const cand = engine.selectCandidate(this.providers, this.strategy, this.rr, req.model, tried, now);
      if (!cand) {
        const w = engine.soonestWait(this.providers, now);
        if (this.wait && w !== null && w <= this.maxWait && (deadline === null || now + w < deadline)) {
          await sleep((Math.min(w, this.maxWait) + 0.01) * 1000);
          tried = engine.forgetRecovered(this.providers, tried, nowS());
          continue;
        }
        break;
      }

      tried.add(engine.triedKey(cand));
      if (!cand.key.reserve(now)) continue;

      try {
        const resp = await this.doRequest(cand, req);
        engine.applySuccess(cand, resp.latencyMs);
        return resp;
      } catch (e) {
        if (e instanceof ProviderError) {
          engine.applyError(cand, e, nowS());
          attempts.push([cand, e]);
          if (engine.shouldRaise(e)) throw e;
          continue;
        }
        throw e;
      }
    }
    throw new NoProvidersAvailable(attempts);
  }

  async text(messages: MessageLike | MessageLike[], opts: ChatOptions = {}): Promise<string> {
    return (await this.chat(messages, opts)).text;
  }

  private async doRequest(cand: Candidate, req: ChatRequest): Promise<ChatResponse> {
    const p = cand.provider;
    const body = buildPayload(req, cand.model);
    const t0 = nowS();
    let res: Response;
    try {
      res = await this.fetchWithTimeout(p.url, {
        method: "POST",
        headers: { ...p.headers(cand.key.key), "User-Agent": UA },
        body: JSON.stringify(body),
      });
    } catch (e: any) {
      throw new Transient(p.name, 0, `transport: ${e?.message ?? e}`);
    }
    const dt = (nowS() - t0) * 1000;
    if (res.status === 200) return p.parseResponse(await res.json(), dt);
    const text = await res.text();
    const err = classify(res.status, headersObj(res), text, p.name);
    if (err instanceof RateLimited) err.scope = p.rateLimitScope(text);
    throw err;
  }

  async *stream(messages: MessageLike | MessageLike[], opts: ChatOptions = {}): AsyncGenerator<string> {
    await this.ensureDiscovered();
    const { model = "auto", ...params } = opts;
    const req = buildRequest(messages, model, params);
    const deadline = this.timeout ? nowS() + this.timeout : null;
    const attempts: Array<[Candidate, Error]> = [];
    let tried = new Set<string>();

    while (attempts.length < this.maxAttempts) {
      const now = nowS();
      if (deadline !== null && now >= deadline) break;
      const cand = engine.selectCandidate(this.providers, this.strategy, this.rr, req.model, tried, now);
      if (!cand) {
        const w = engine.soonestWait(this.providers, now);
        if (this.wait && w !== null && w <= this.maxWait && (deadline === null || now + w < deadline)) {
          await sleep((Math.min(w, this.maxWait) + 0.01) * 1000);
          tried = engine.forgetRecovered(this.providers, tried, nowS());
          continue;
        }
        break;
      }

      tried.add(engine.triedKey(cand));
      if (!cand.key.reserve(now)) continue;

      let produced = false;
      try {
        for await (const chunk of this.streamRequest(cand, req)) {
          produced = true;
          yield chunk;
        }
      } catch (e) {
        if (e instanceof ProviderError) {
          engine.applyError(cand, e, nowS());
          attempts.push([cand, e]);
          if (produced || engine.shouldRaise(e)) throw e;
          continue;
        }
        throw e;
      }
      engine.applySuccess(cand, 0);
      return;
    }
    throw new NoProvidersAvailable(attempts);
  }

  private async *streamRequest(cand: Candidate, req: ChatRequest): AsyncGenerator<string> {
    const p = cand.provider;
    const body = { ...buildPayload(req, cand.model), stream: true };
    let res: Response;
    try {
      res = await this.fetchWithTimeout(p.url, {
        method: "POST",
        headers: { ...p.headers(cand.key.key), "User-Agent": UA },
        body: JSON.stringify(body),
      });
    } catch (e: any) {
      throw new Transient(p.name, 0, `transport: ${e?.message ?? e}`);
    }
    if (res.status !== 200) {
      const text = await res.text();
      const err = classify(res.status, headersObj(res), text, p.name);
      if (err instanceof RateLimited) err.scope = p.rateLimitScope(text);
      throw err;
    }
    if (!res.body) return;

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl: number;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          const delta = sseDelta(line);
          if (delta) yield delta;
        }
      }
      const delta = sseDelta(buf.trim());
      if (delta) yield delta;
    } finally {
      try {
        reader.releaseLock();
      } catch {
        /* ignore */
      }
    }
  }
}
