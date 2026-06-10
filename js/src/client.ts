/** The user-facing async client: FreeLLM (chat, text, stream, health). */
import { discover } from "./discovery.js";
import * as engine from "./engine.js";
import { ConfigError, NoProvidersAvailable, ProviderError, RateLimited, Transient, classify } from "./errors.js";
import { Provider } from "./providers/base.js";
import { providersFromEnv } from "./config.js";
import { StateStore } from "./state.js";
import { Candidate, STRATEGIES, Strategy } from "./strategy.js";
import { nowS, sleep } from "./time.js";
import { buildPayload, buildRequest, ChatRequest, ChatResponse, FreeLLMEvent, MessageLike } from "./types.js";
import { VERSION } from "./version.js";

const UA = `freelm-js/${VERSION}`;

export interface FreeLLMOptions {
  strategy?: Strategy;
  maxAttempts?: number;
  timeout?: number; // seconds; also the overall per-call deadline
  wait?: boolean;
  maxWait?: number;
  onEvent?: (e: FreeLLMEvent) => void;
  /** Persist rpd counters / cooldowns / disabled keys across restarts
   * (~/.cache/freelm/state.json). Defaults to the FREELM_PERSIST env var. */
  persist?: boolean;
}

export type ChatOptions = { model?: string | string[] } & Record<string, any>;

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
  private onEvent?: (e: FreeLLMEvent) => void;
  private state: StateStore | null = null;

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
    this.onEvent = opts.onEvent;
    const persist = opts.persist ?? ["1", "true", "yes"].includes((process.env.FREELM_PERSIST ?? "").toLowerCase());
    if (persist) {
      this.state = new StateStore();
      this.state.loadInto(this.providers, nowS());
    }
  }

  private emit(
    kind: FreeLLMEvent["kind"],
    extra: { cand?: Candidate; provider?: string; status?: number; latencyMs?: number; error?: string; attempt?: number } = {},
  ): void {
    if (!this.onEvent) return;
    try {
      this.onEvent({
        kind,
        provider: extra.cand?.provider.name ?? extra.provider ?? null,
        key: extra.cand?.key.masked() ?? null,
        model: extra.cand?.model ?? null,
        status: extra.status ?? null,
        latencyMs: extra.latencyMs ?? null,
        error: extra.error ?? null,
        attempt: extra.attempt ?? 0,
      });
    } catch {
      // a misbehaving callback must never break the call
    }
  }

  private saveState(): void {
    this.state?.save(this.providers, nowS());
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
            if (await discover(p)) this.emit("discovery", { provider: p.name });
          } catch {
            /* keep fallback models */
          }
        }
      }),
    );
    this.discoveryDone = true;
  }

  /** AbortController armed with `timeout`; the caller re-arms it per phase
   * (headers, body read, each stream chunk) and MUST call done() at the end —
   * unlike httpx, fetch stops honouring a cleared timer once headers arrive,
   * so a stalled body would otherwise hang forever. */
  private timer(): { signal: AbortSignal | undefined; rearm: () => void; done: () => void } {
    if (!this.timeout) return { signal: undefined, rearm: () => {}, done: () => {} };
    const ac = new AbortController();
    let t = setTimeout(() => ac.abort(), this.timeout * 1000);
    return {
      signal: ac.signal,
      rearm: () => {
        clearTimeout(t);
        t = setTimeout(() => ac.abort(), this.timeout * 1000);
      },
      done: () => clearTimeout(t),
    };
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
          this.emit("wait", { latencyMs: w * 1000, attempt: attempts.length });
          await sleep((Math.min(w, this.maxWait) + 0.01) * 1000);
          tried = engine.forgetRecovered(this.providers, tried, nowS());
          continue;
        }
        break;
      }

      tried.add(engine.triedKey(cand));
      if (!cand.key.reserve(now)) continue;

      this.emit("attempt", { cand, attempt: attempts.length + 1 });
      try {
        const resp = await this.doRequest(cand, req);
        engine.applySuccess(cand, resp.latencyMs);
        this.emit("success", { cand, latencyMs: resp.latencyMs, attempt: attempts.length + 1 });
        this.saveState();
        return resp;
      } catch (e) {
        if (e instanceof ProviderError) {
          engine.applyError(cand, e, nowS());
          attempts.push([cand, e]);
          this.emit("error", { cand, status: e.status, error: String(e.message), attempt: attempts.length });
          this.saveState();
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
    const { signal, rearm, done } = this.timer();
    try {
      let res: Response;
      try {
        res = await fetch(p.url, {
          method: "POST",
          headers: { ...p.headers(cand.key.key), "User-Agent": UA },
          body: JSON.stringify(body),
          signal,
        });
      } catch (e: any) {
        throw new Transient(p.name, 0, `transport: ${e?.message ?? e}`);
      }
      rearm(); // body read gets its own timeout window
      const dt = (nowS() - t0) * 1000;
      if (res.status === 200) {
        let data: any;
        try {
          data = await res.json();
        } catch (e: any) {
          throw new Transient(p.name, 0, `read: ${e?.message ?? e}`);
        }
        return p.parseResponse(data, dt);
      }
      const text = await res.text().catch(() => "");
      const err = classify(res.status, headersObj(res), text, p.name);
      if (err instanceof RateLimited) err.scope = p.rateLimitScope(text);
      throw err;
    } finally {
      done();
    }
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
          this.emit("wait", { latencyMs: w * 1000, attempt: attempts.length });
          await sleep((Math.min(w, this.maxWait) + 0.01) * 1000);
          tried = engine.forgetRecovered(this.providers, tried, nowS());
          continue;
        }
        break;
      }

      tried.add(engine.triedKey(cand));
      if (!cand.key.reserve(now)) continue;

      let produced = false;
      let firstMs = 0; // time-to-first-token; feeds the latency EWMA
      const t0 = nowS();
      this.emit("attempt", { cand, attempt: attempts.length + 1 });
      try {
        for await (const chunk of this.streamRequest(cand, req)) {
          if (!produced) firstMs = (nowS() - t0) * 1000;
          produced = true;
          yield chunk;
        }
      } catch (e) {
        if (e instanceof ProviderError) {
          engine.applyError(cand, e, nowS());
          attempts.push([cand, e]);
          this.emit("error", { cand, status: e.status, error: String(e.message), attempt: attempts.length });
          this.saveState();
          if (produced || engine.shouldRaise(e)) throw e;
          continue;
        }
        throw e;
      }
      engine.applySuccess(cand, firstMs);
      this.emit("success", { cand, latencyMs: firstMs, attempt: attempts.length + 1 });
      this.saveState();
      return;
    }
    throw new NoProvidersAvailable(attempts);
  }

  private async *streamRequest(cand: Candidate, req: ChatRequest): AsyncGenerator<string> {
    const p = cand.provider;
    const body = { ...buildPayload(req, cand.model), stream: true };
    const { signal, rearm, done: clearTimer } = this.timer();
    let res: Response;
    try {
      res = await fetch(p.url, {
        method: "POST",
        headers: { ...p.headers(cand.key.key), "User-Agent": UA },
        body: JSON.stringify(body),
        signal,
      });
    } catch (e: any) {
      clearTimer();
      throw new Transient(p.name, 0, `transport: ${e?.message ?? e}`);
    }
    if (res.status !== 200) {
      try {
        const text = await res.text().catch(() => "");
        const err = classify(res.status, headersObj(res), text, p.name);
        if (err instanceof RateLimited) err.scope = p.rateLimitScope(text);
        throw err;
      } finally {
        clearTimer();
      }
    }
    if (!res.body) {
      clearTimer();
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    try {
      for (;;) {
        rearm(); // inactivity timeout per chunk (mirrors httpx's read timeout)
        let chunk: ReadableStreamReadResult<Uint8Array>;
        try {
          chunk = await reader.read();
        } catch (e: any) {
          throw new Transient(p.name, 0, `read: ${e?.message ?? e}`);
        }
        if (chunk.done) break;
        buf += decoder.decode(chunk.value, { stream: true });
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
      clearTimer();
      try {
        await reader.cancel(); // close the connection if the consumer bailed early
      } catch {
        /* ignore */
      }
      try {
        reader.releaseLock();
      } catch {
        /* ignore */
      }
    }
  }
}
