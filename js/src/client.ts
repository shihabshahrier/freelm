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
  private rr = { p: 0 };
  private discovering: Promise<void> | null = null;
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
    engine.applySuccess(cand, latencyMs);
    this.emit("success", { cand, latencyMs, attempt: attempts.length + 1 });
    this.saveState();
  }

  async chat(messages: MessageLike | MessageLike[], opts: ChatOptions = {}): Promise<ChatResponse> {
    const { model = "auto", signal, stream, ...params } = opts;
    if (stream) throw new ConfigError("chat() returns a whole response; use stream() / streamChunks() to stream");
    throwIfAborted(signal);
    await this.ensureDiscovered();
    const req = buildRequest(messages, model, params);
    const deadline = this.timeout ? nowS() + this.timeout : null;
    const attempts: Array<[Candidate, Error]> = [];
    let tried = new Set<string>();

    while (attempts.length < this.maxAttempts) {
      throwIfAborted(signal);
      const now = nowS();
      if (deadline !== null && now >= deadline) break;
      const cand = engine.selectCandidate(
        this.providers, this.strategy, this.rr, req.model, tried, now, engine.rejectedBy(attempts),
      );
      if (!cand) {
        const w = this.waitFor(now, deadline, req.model, tried, attempts);
        if (w === null) break;
        this.emit("wait", { latencyMs: w * 1000, attempt: attempts.length });
        await sleep((w + 0.01) * 1000, signal);
        tried = engine.forgetRecovered(this.providers, tried, nowS());
        continue;
      }

      tried.add(engine.triedKey(cand));
      if (!cand.key.reserve(now)) continue;

      this.emit("attempt", { cand, attempt: attempts.length + 1 });
      let resp: ChatResponse;
      try {
        resp = await this.doRequest(cand, req, deadline, signal);
      } catch (e) {
        if (!(e instanceof ProviderError)) throw e; // e.g. the caller aborted
        this.recordError(cand, e, attempts);
        if (engine.shouldRaise(e, attempts)) throw e;
        continue;
      }
      this.recordSuccess(cand, resp.latencyMs, attempts);
      return resp;
    }
    throw engine.exhausted(attempts, this.providers, nowS());
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
    const attempts: Array<[Candidate, Error]> = [];
    let tried = new Set<string>();

    while (attempts.length < this.maxAttempts) {
      throwIfAborted(signal);
      const now = nowS();
      if (deadline !== null && now >= deadline) break;
      const cand = engine.selectCandidate(
        this.providers, this.strategy, this.rr, req.model, tried, now, engine.rejectedBy(attempts),
      );
      if (!cand) {
        const w = this.waitFor(now, deadline, req.model, tried, attempts);
        if (w === null) break;
        this.emit("wait", { latencyMs: w * 1000, attempt: attempts.length });
        await sleep((w + 0.01) * 1000, signal);
        tried = engine.forgetRecovered(this.providers, tried, nowS());
        continue;
      }

      tried.add(engine.triedKey(cand));
      if (!cand.key.reserve(now)) continue;

      let produced = false;
      let firstMs = 0; // time-to-first-token; feeds the latency EWMA
      let pending: any[] = []; // raw mode: chunks held until one carries output
      const t0 = nowS();
      this.emit("attempt", { cand, attempt: attempts.length + 1 });
      try {
        for await (const chunk of this.streamRequest(cand, req, signal)) {
          let out: any[];
          if (raw) {
            if (!produced && !hasOutput(chunk)) {
              pending.push(chunk);
              continue;
            }
            out = [...pending, chunk];
            pending = [];
          } else {
            const t = chunkText(chunk);
            if (!t) continue;
            out = [t];
          }
          if (!produced) {
            firstMs = (nowS() - t0) * 1000;
            produced = true;
          }
          for (const o of out) yield o;
        }
      } catch (e) {
        if (!(e instanceof ProviderError)) throw e; // e.g. the caller aborted
        this.recordError(cand, e, attempts);
        if (produced || engine.shouldRaise(e, attempts)) throw e;
        continue;
      }
      for (const o of pending) yield o; // a raw stream that never carried output (empty completion)
      this.recordSuccess(cand, firstMs, attempts);
      return;
    }
    throw engine.exhausted(attempts, this.providers, nowS());
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
