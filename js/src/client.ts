/** The user-facing async client: FreeLLM (chat, text, stream, streamChunks, health). */
import { CircuitBreaker } from "./breaker.js";
import * as cache from "./cache.js";
import { KeylessArg, providersFromEnv } from "./config.js";
import { discover } from "./discovery.js";
import * as engine from "./engine.js";
import { ConfigError, ProviderError, RateLimited, Transient, classify } from "./errors.js";
import { Provider } from "./providers/base.js";
import { env, unref } from "./runtime.js";
import { StateStore } from "./state.js";
import { Candidate, STRATEGIES, Strategy } from "./strategy.js";
import { abortError, nowS, sleep } from "./time.js";
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
  /** Race a slow attempt: when it is still running after the hedge delay, the
   * next candidate starts in parallel and the first answer wins. `true`
   * (default) = adaptive delay, a number = fixed seconds, `false` = sequential. */
  hedge?: boolean | number;
}

/** One in-flight attempt of a call. */
interface Run<T> {
  cand: Candidate;
  t0: number;
  ac: AbortController;
  unlink: () => void;
  done: Promise<Settled<T>>;
}
type Settled<T> = { run: Run<T>; ok: true; value: T } | { run: Run<T>; ok: false; error: unknown };

/** A stream read up to its first emittable item. */
interface Opened {
  gen: AsyncGenerator<Record<string, any>>;
  items: any[];
  done: boolean;
  firstMs: number;
}

/** The first of `promises` to settle, or null after `ms` (null = no limit). */
async function raceTimeout<T>(promises: Promise<T>[], ms: number | null): Promise<T | null> {
  if (ms === null) return Promise.race(promises);
  let timer: any;
  const timeout = new Promise<null>((resolve) => {
    timer = setTimeout(() => resolve(null), ms);
    unref(timer);
  });
  try {
    return await Promise.race([...promises, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

/** Per-call options: `model` (alias, concrete id, or ordered fallback list),
 * `signal` (an AbortSignal that cancels the call), plus OpenAI request fields
 * (temperature, max_tokens, tools, response_format, ...) passed through. */
export type ChatOptions = { model?: string | string[]; signal?: AbortSignal } & Record<string, any>;

function headersObj(res: Response): Record<string, string> {
  const o: Record<string, string> = {};
  res.headers.forEach((v, k) => (o[k.toLowerCase()] = v));
  return o;
}

/** Split streamed text into lines on "\n", "\r\n" or a lone "\r". */
class LineSplitter {
  private buf = "";

  push(text: string): string[] {
    this.buf += text;
    const out: string[] = [];
    for (;;) {
      const m = /\r\n|\r|\n/.exec(this.buf);
      if (!m) break;
      // a trailing "\r" may be the first half of "\r\n" — wait for more bytes
      if (m[0] === "\r" && m.index === this.buf.length - 1) break;
      out.push(this.buf.slice(0, m.index));
      this.buf = this.buf.slice(m.index + m[0].length);
    }
    return out;
  }

  flush(): string[] {
    const rest = this.buf;
    this.buf = "";
    return rest ? [rest.replace(/\r$/, "")] : [];
  }
}

function loads(text: string): any {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

/** Incremental decoder for OpenAI-style SSE: one JSON object per `data:` event.
 * Handles comments/`event:` lines, `[DONE]`, multi-line `data:` events, and
 * servers that skip the blank separator line. */
class SSE {
  private buf: string[] = [];
  done = false;

  feed(line: string): Record<string, any> | null {
    if (!line) {
      this.buf = []; // blank line = end of event
      return null;
    }
    if (!line.startsWith("data:")) return null; // ": keep-alive", event:, id:, retry:
    let data = line.slice(5);
    if (data.startsWith(" ")) data = data.slice(1);
    if (!this.buf.length && data.trim() === "[DONE]") {
      this.done = true;
      return null;
    }
    this.buf.push(data);
    let obj = loads(this.buf.join("\n"));
    if (obj === undefined && this.buf.length > 1) obj = loads(data); // a stale fragment must not swallow a valid line
    if (obj === undefined) return null; // incomplete multi-line event: wait for more
    this.buf = [];
    return obj && typeof obj === "object" && !Array.isArray(obj) ? obj : null;
  }
}

/** The text delta of a chat.completion.chunk (content-part arrays flattened), or null. */
function chunkText(chunk: any): string | null {
  const choices = Array.isArray(chunk?.choices) ? chunk.choices : [];
  let c = choices[0]?.delta?.content;
  if (Array.isArray(c)) c = c.filter((p: any) => p && p.type === "text").map((p: any) => p.text ?? "").join("");
  return typeof c === "string" && c ? c : null;
}

/** Workers AI sometimes streams a numeric token as a JSON number
 * (`"content": 6`); make it text before anyone joins the deltas. */
function repairChunk(chunk: any): void {
  for (const c of Array.isArray(chunk?.choices) ? chunk.choices : []) {
    const d = c?.delta;
    if (d && typeof d.content === "number") d.content = String(d.content);
  }
}

/** Does this chunk carry anything a consumer would act on (text, tool calls,
 * reasoning, a finish reason)? Role-only preambles don't. */
function hasOutput(chunk: any): boolean {
  for (const c of Array.isArray(chunk?.choices) ? chunk.choices : []) {
    if (!c || typeof c !== "object") continue;
    if (c.finish_reason) return true;
    const d = c.delta ?? {};
    if (d.content || d.tool_calls || d.reasoning || d.reasoning_content) return true;
  }
  return false;
}

/** classify() plus the provider's say on whether the error is key- or model-scoped. */
function classifyFor(p: Provider, status: number, headers: Record<string, string> | null, body: string): ProviderError {
  const err = classify(status, headers, body, p.name);
  if (err instanceof RateLimited) err.scope = p.rateLimitScope(body);
  else if (err instanceof Transient) err.scope = p.transientScope(body);
  return err;
}

/** An {"error": ...} object delivered with HTTP 200 (whole body or a mid-stream
 * SSE frame). Uses its numeric `code` when present; else an upstream failure (502). */
function errorFrame(p: Provider, err: any): ProviderError {
  const code = err && typeof err === "object" ? err.code : null;
  const status = typeof code === "number" && code >= 400 && code < 600 ? code : 502;
  const body = typeof err === "string" ? err : JSON.stringify({ error: err });
  return classifyFor(p, status, null, body);
}

function parseOk(p: Provider, text: string, latencyMs: number): ChatResponse {
  const data = loads(text);
  if (data === undefined) throw new Transient(p.name, 200, "invalid JSON in response body");
  if (!data || typeof data !== "object" || Array.isArray(data)) throw new Transient(p.name, 200, "unexpected response shape");
  if (data.error) throw errorFrame(p, data.error);
  return p.parseResponse(data, latencyMs);
}

/** Abort `ac` when the caller's signal fires; returns an unlink function. */
function link(signal: AbortSignal | undefined, ac: AbortController): () => void {
  if (!signal) return () => {};
  if (signal.aborted) {
    ac.abort();
    return () => {};
  }
  const onAbort = () => ac.abort();
  signal.addEventListener("abort", onAbort, { once: true });
  return () => signal.removeEventListener("abort", onAbort);
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw abortError(signal);
}

export class FreeLLM {
  providers: Provider[];
  strategy: string;
  maxAttempts: number;
  timeout: number;
  wait: boolean;
  maxWait: number;
  hedge: boolean | number;
  private rr = { p: 0 };
  private discovering: Promise<void> | null = null;
  private onEvent?: (e: FreeLLMEvent) => void;
  private state: StateStore | null = null;

  constructor(providers: Provider[], opts: FreeLLMOptions = {}) {
    if (!providers.length) throw new ConfigError("FreeLLM needs at least one provider");
    this.strategy = opts.strategy ?? "smart";
    if (!STRATEGIES.includes(this.strategy as Strategy)) {
      throw new ConfigError(`unknown strategy ${this.strategy}; pick one of ${STRATEGIES.join(", ")}`);
    }
    this.providers = providers;
    this.maxAttempts = opts.maxAttempts ?? 12;
    this.timeout = opts.timeout ?? 60;
    this.wait = opts.wait ?? false;
    this.maxWait = opts.maxWait ?? 20;
    this.hedge = opts.hedge ?? true;
    if (typeof this.hedge !== "boolean" && !(typeof this.hedge === "number" && this.hedge >= 0)) {
      throw new ConfigError("hedge must be true (adaptive), a delay in seconds, or false");
    }
    this.onEvent = opts.onEvent;
    const persist = opts.persist ?? ["1", "true", "yes"].includes((env("FREELM_PERSIST") ?? "").toLowerCase());
    if (persist) {
      this.state = new StateStore();
      this.state.loadInto(this.providers, nowS());
    }
  }

  /** Providers from environment keys; `keyless: true` / "auto" adds the
   * no-signup endpoints (always / only when no keys are set). */
  static fromEnv(opts: FreeLLMOptions & { keyless?: KeylessArg } = {}): FreeLLM {
    const { keyless, ...rest } = opts;
    return new FreeLLM(providersFromEnv(keyless), rest);
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

  /** Give every key and model a clean slate: re-enable disabled keys, clear
   * cooldowns, breakers and benched models (e.g. after fixing a key). */
  resetKeys(): void {
    for (const p of this.providers) {
      p._modelUntil.clear();
      for (const k of p.keys) {
        k.modelUntil.clear();
        k.disabled = false;
        k.disabledSinceWall = 0;
        k.cooldownUntil = 0;
        k.breaker = new CircuitBreaker();
        k.lastError = null;
      }
    }
    this.saveState();
  }

  /** Force a live re-discovery on the next call (bypasses the in-memory guard
   * and the disk cache). */
  refreshModels(): void {
    this.discovering = null;
    for (const p of this.providers) {
      p._discovered = false;
      if (p.discover) cache.clear(p.name);
    }
  }

  /** One discovery for N concurrent first calls. */
  private ensureDiscovered(): Promise<void> {
    this.discovering ??= Promise.all(
      this.providers
        .filter((p) => p.discover)
        .map(async (p) => {
          try {
            if (await discover(p)) this.emit("discovery", { provider: p.name });
          } catch {
            /* keep fallback models */
          }
        }),
    ).then(() => undefined);
    return this.discovering;
  }

  /** Seconds to sleep when no candidate is ready, or null to give up. Zero means
   * keys are ready but every candidate was tried or benched — waiting can't help. */
  private waitFor(now: number, deadline: number | null, alias: string | string[], tried: Set<string>, attempts: Array<[Candidate, Error]>): number | null {
    if (!this.wait) return null;
    const w = engine.soonestWait(this.providers, now, alias, tried, engine.rejectedBy(attempts));
    if (w === null || w <= 0 || w > this.maxWait) return null;
    if (deadline !== null && now + w >= deadline) return null;
    return w;
  }

  private recordError(cand: Candidate, e: ProviderError, attempts: Array<[Candidate, Error]>): void {
    engine.applyError(cand, e, nowS());
    attempts.push([cand, e]);
    this.emit("error", { cand, status: e.status, error: String(e.message), attempt: attempts.length });
    this.saveState();
  }

  private recordSuccess(cand: Candidate, latencyMs: number, attempts: Array<[Candidate, Error]>): void {
    engine.applySuccess(cand, latencyMs, nowS());
    this.emit("success", { cand, latencyMs, attempt: attempts.length + 1 });
    this.saveState();
  }

  /** The next candidate to start (its rpm token reserved), or null. */
  private pick(req: ChatRequest, tried: Set<string>, attempts: Array<[Candidate, Error]>, running: number, now: number): Candidate | null {
    while (attempts.length + running < this.maxAttempts) {
      const cand = engine.selectCandidate(this.providers, this.strategy, this.rr, req.model, tried, now, engine.rejectedBy(attempts));
      if (!cand) return null;
      tried.add(engine.triedKey(cand));
      if (cand.key.reserve(now)) return cand;
    }
    return null;
  }

  /** When the single running attempt gets a parallel hedge (monotonic s), or null. */
  private hedgeAt(running: Array<{ cand: Candidate; t0: number }>, stream: boolean): number | null {
    if (running.length !== 1) return null;
    const d = engine.hedgeDelay(running[0].cand, this.hedge, stream);
    return d === null ? null : running[0].t0 + d;
  }

  /** Run attempts until one succeeds. One at a time; when it is still running
   * after the hedge delay (`hedge`), the next candidate starts in parallel and
   * the first answer wins — the others are aborted. `work(cand, signal)` does
   * the HTTP; `cleanup(result)` releases a result that lost the race (an open
   * stream). `release()` unlinks the caller's signal from the winner. */
  private async race<T>(
    req: ChatRequest,
    deadline: number | null,
    signal: AbortSignal | undefined,
    stream: boolean,
    work: (cand: Candidate, signal: AbortSignal) => Promise<T>,
    cleanup?: (result: T) => unknown,
  ): Promise<{ cand: Candidate; result: T; attempts: Array<[Candidate, Error]>; release: () => void }> {
    const attempts: Array<[Candidate, Error]> = [];
    let tried = new Set<string>();
    const running: Run<T>[] = [];
    let blocked = false; // a hedge was due but no candidate was ready
    const start = (cand: Candidate): Run<T> => {
      const ac = new AbortController();
      const run = { cand, t0: nowS(), ac, unlink: link(signal, ac) } as Run<T>;
      run.done = work(cand, ac.signal).then(
        (value) => ({ run, ok: true as const, value }),
        (error) => ({ run, ok: false as const, error }),
      );
      return run;
    };
    // Leave attempts that lost the race (or outlived the call). One that
    // started before the winner was slower than it: remember that.
    const abandon = (winnerT0: number | null) => {
      const now = nowS();
      for (const r of running.splice(0)) {
        if (winnerT0 !== null && r.t0 < winnerT0) engine.applySlow(r.cand, (now - r.t0) * 1000, now);
        r.ac.abort();
        r.unlink();
        void r.done.then((s) => {
          if (!s.ok || !cleanup) return;
          try {
            Promise.resolve(cleanup(s.value)).catch(() => {});
          } catch {
            /* best effort */
          }
        });
      }
    };
    try {
      for (;;) {
        throwIfAborted(signal);
        const now = nowS();
        if (deadline !== null && now >= deadline) break;
        let hedgeAt = blocked ? null : this.hedgeAt(running, stream);
        if (!running.length || (hedgeAt !== null && now >= hedgeAt)) {
          const cand = this.pick(req, tried, attempts, running.length, now);
          if (cand) {
            this.emit(running.length ? "hedge" : "attempt", { cand, attempt: attempts.length + running.length + 1 });
            running.push(start(cand));
            continue;
          }
          if (!running.length) {
            const w = this.waitFor(now, deadline, req.model, tried, attempts);
            if (w === null) break;
            this.emit("wait", { latencyMs: w * 1000, attempt: attempts.length });
            await sleep((w + 0.01) * 1000, signal);
            tried = engine.forgetRecovered(this.providers, tried, nowS());
            continue;
          }
          blocked = true;
          hedgeAt = null;
        }
        const wake = [deadline, hedgeAt].filter((t): t is number => t !== null);
        const settled = await raceTimeout(
          running.map((r) => r.done),
          wake.length ? Math.max(0, (Math.min(...wake) - now) * 1000) : null,
        );
        if (!settled) continue;
        running.splice(running.indexOf(settled.run), 1);
        blocked = false;
        if (settled.ok) {
          abandon(settled.run.t0);
          return { cand: settled.run.cand, result: settled.value, attempts, release: settled.run.unlink };
        }
        settled.run.unlink();
        const e = settled.error;
        if (!(e instanceof ProviderError)) throw e; // e.g. the caller aborted
        this.recordError(settled.run.cand, e, attempts);
        if (engine.shouldRaise(e, attempts)) throw e;
      }
      for (const r of running) {
        this.recordError(r.cand, new Transient(r.cand.provider.name, 0, "timeout: no answer within the call's deadline"), attempts);
      }
      throw engine.exhausted(attempts, this.providers, nowS());
    } finally {
      abandon(null);
    }
  }

  async chat(messages: MessageLike | MessageLike[], opts: ChatOptions = {}): Promise<ChatResponse> {
    const { model = "auto", signal, stream, ...params } = opts;
    if (stream) throw new ConfigError("chat() returns a whole response; use stream() / streamChunks() to stream");
    throwIfAborted(signal);
    await this.ensureDiscovered();
    const req = buildRequest(messages, model, params);
    const deadline = this.timeout ? nowS() + this.timeout : null;
    const { cand, result, attempts, release } = await this.race(req, deadline, signal, false, (c, s) =>
      this.doRequest(c, req, deadline, s),
    );
    release();
    this.recordSuccess(cand, result.latencyMs, attempts);
    return result;
  }

  async text(messages: MessageLike | MessageLike[], opts: ChatOptions = {}): Promise<string> {
    return (await this.chat(messages, opts)).text;
  }

  private async doRequest(cand: Candidate, req: ChatRequest, deadline: number | null, signal?: AbortSignal): Promise<ChatResponse> {
    const p = cand.provider;
    const body = p.adaptPayload(buildPayload(req, cand.model));
    const t0 = nowS();
    // one attempt may not outlive the call's overall deadline; the timer spans
    // headers *and* body (fetch doesn't bound body reads on its own)
    const ms = deadline !== null ? Math.max(100, (deadline - nowS()) * 1000) : this.timeout ? this.timeout * 1000 : null;
    const ac = new AbortController();
    const unlink = link(signal, ac);
    const timer = ms !== null ? setTimeout(() => ac.abort(), ms) : null;
    unref(timer);
    try {
      let res: Response;
      try {
        res = await fetch(p.url, {
          method: "POST",
          headers: { ...p.headers(cand.key.key), "User-Agent": UA },
          body: JSON.stringify(body),
          signal: ac.signal,
        });
      } catch (e: any) {
        if (signal?.aborted) throw abortError(signal);
        throw new Transient(p.name, 0, ac.signal.aborted ? "timeout" : `transport: ${e?.message ?? e}`);
      }
      let text: string;
      try {
        text = await res.text();
      } catch (e: any) {
        if (signal?.aborted) throw abortError(signal);
        throw new Transient(p.name, res.status === 200 ? 0 : res.status, `read: ${e?.message ?? e}`);
      }
      const dt = (nowS() - t0) * 1000;
      if (res.status === 200) return parseOk(p, text, dt);
      throw classifyFor(p, res.status, headersObj(res), text);
    } finally {
      if (timer) clearTimeout(timer);
      unlink();
    }
  }

  /** Yield content deltas as they arrive. Fails over between providers *before*
   * the first token; once tokens flow it stays on that provider. */
  stream(messages: MessageLike | MessageLike[], opts: ChatOptions = {}): AsyncGenerator<string> {
    return this.streamLoop(messages, opts, false);
  }

  /** Like stream(), but yield the raw OpenAI `chat.completion.chunk` objects —
   * tool-call deltas, finish reasons and usage included. Chunks carrying no
   * output yet (a role-only preamble) are held back until the first real one,
   * so failover stays invisible to the consumer. */
  streamChunks(messages: MessageLike | MessageLike[], opts: ChatOptions = {}): AsyncGenerator<Record<string, any>> {
    return this.streamLoop(messages, opts, true);
  }

  private async *streamLoop(messages: MessageLike | MessageLike[], opts: ChatOptions, raw: boolean): AsyncGenerator<any> {
    const { model = "auto", signal, stream: _stream, ...params } = opts;
    throwIfAborted(signal);
    await this.ensureDiscovered();
    const req = buildRequest(messages, model, params);
    const deadline = this.timeout ? nowS() + this.timeout : null;
    const { cand, result, attempts, release } = await this.race(
      req,
      deadline,
      signal,
      true,
      (c, s) => this.openStream(c, req, raw, s),
      (o) => o.gen.return(undefined),
    );
    const { gen, items, done, firstMs } = result;
    try {
      for (const o of items) yield o;
      if (!done) {
        for (;;) {
          const r = await gen.next();
          if (r.done) break;
          if (raw) yield r.value;
          else {
            const t = chunkText(r.value);
            if (t) yield t;
          }
        }
      }
    } catch (e) {
      if (e instanceof ProviderError) this.recordError(cand, e, attempts);
      throw e; // output already reached the caller: no mid-stream failover
    } finally {
      await gen.return(undefined);
      release();
    }
    this.recordSuccess(cand, firstMs, attempts);
  }

  /** Start a stream and read up to its first emittable item: `items` go out
   * first (raw mode: the held-back role-only preamble plus the first output
   * chunk); `done` means it ended without output (an empty completion). */
  private async openStream(cand: Candidate, req: ChatRequest, raw: boolean, signal: AbortSignal): Promise<Opened> {
    const t0 = nowS();
    const gen = this.streamRequest(cand, req, signal);
    const pending: any[] = [];
    try {
      for (;;) {
        const r = await gen.next();
        if (r.done) return { gen, items: pending, done: true, firstMs: 0 };
        const chunk = r.value;
        if (raw) {
          if (!hasOutput(chunk)) {
            pending.push(chunk);
            continue;
          }
          return { gen, items: [...pending, chunk], done: false, firstMs: (nowS() - t0) * 1000 };
        }
        const t = chunkText(chunk);
        if (t) return { gen, items: [t], done: false, firstMs: (nowS() - t0) * 1000 };
      }
    } catch (e) {
      await gen.return(undefined).catch(() => {});
      throw e;
    }
  }

  /** Read one chunk with an inactivity timeout. The timer runs only while we
   * wait on the network — never while the consumer holds a yielded chunk — and
   * races the read itself, since an aborted fetch doesn't always settle it. */
  private async readChunk(
    reader: ReadableStreamDefaultReader<Uint8Array>,
    p: Provider,
    signal?: AbortSignal,
  ): Promise<ReadableStreamReadResult<Uint8Array>> {
    let timer: any = null;
    const read = reader.read();
    const stalled = this.timeout
      ? new Promise<never>((_, reject) => {
          timer = setTimeout(() => reject(new Transient(p.name, 0, `stream stalled for ${this.timeout}s`)), this.timeout * 1000);
          unref(timer);
        })
      : null;
    try {
      return await (stalled ? Promise.race([read, stalled]) : read);
    } catch (e: any) {
      if (signal?.aborted) throw abortError(signal);
      if (e instanceof Transient) throw e;
      throw new Transient(p.name, 0, `read: ${e?.message ?? e}`);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  private async *streamRequest(cand: Candidate, req: ChatRequest, signal?: AbortSignal): AsyncGenerator<Record<string, any>> {
    const p = cand.provider;
    const body = p.adaptPayload({ ...buildPayload(req, cand.model), stream: true });
    const ac = new AbortController();
    const unlink = link(signal, ac);
    let res: Response;
    const headersTimer = this.timeout ? setTimeout(() => ac.abort(), this.timeout * 1000) : null;
    unref(headersTimer);
    try {
      try {
        res = await fetch(p.url, {
          method: "POST",
          headers: { ...p.headers(cand.key.key), "User-Agent": UA },
          body: JSON.stringify(body),
          signal: ac.signal,
        });
      } catch (e: any) {
        if (signal?.aborted) throw abortError(signal);
        throw new Transient(p.name, 0, ac.signal.aborted ? "timeout" : `transport: ${e?.message ?? e}`);
      }
      if (res.status !== 200) {
        const text = await res.text().catch(() => "");
        throw classifyFor(p, res.status, headersObj(res), text);
      }
    } catch (e) {
      unlink();
      throw e;
    } finally {
      if (headersTimer) clearTimeout(headersTimer);
    }
    if (!res.body) {
      unlink();
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    const lines = new LineSplitter();
    const sse = new SSE();
    let output = false;
    try {
      for (;;) {
        const r = await this.readChunk(reader, p, signal);
        const batch = r.done ? [...lines.push(decoder.decode()), ...lines.flush()] : lines.push(decoder.decode(r.value, { stream: true }));
        for (const line of batch) {
          const obj = sse.feed(line);
          if (sse.done) return; // [DONE]: stop even if the server keeps the connection open
          if (!obj) continue;
          if (obj.error) throw errorFrame(p, obj.error);
          repairChunk(obj);
          output ||= hasOutput(obj);
          yield obj;
        }
        if (r.done) {
          // an empty or cut-off 200 before any output: fail over
          if (!output) throw new Transient(p.name, 200, "stream ended before any output");
          return;
        }
      }
    } finally {
      unlink();
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
