# Failover simulation

How long does a broken provider cost a call? These scripts put one misbehaving
fake provider in front of a healthy one (both on localhost, over real HTTP),
run freelm **unmodified with its default settings**, and time two calls on the
same client. The first call shows how fast freelm gets past the failure; the
second shows whether it learned from it.

```bash
python benchmarks/failover.py                 # this checkout: sync + async clients
python benchmarks/failover.py --no-hedge      # hedging off (smart routing stays on)
pip install freelm==0.4.0 && python benchmarks/failover.py --installed --sync-only   # a release, for comparison

cd js && npm run build && cd .. && node benchmarks/failover.mjs   # the JS/TS client
node benchmarks/failover.mjs --package freelm                      # an installed npm package
```

No API keys, no network beyond localhost (the "unroutable host" case dials a
non-routable private address on purpose). A full run takes about a minute.

## Results — 2026-10-10

macOS arm64, Python 3.14, Node 22. The healthy provider answers in 0.3 s; times
are wall-clock for one `chat()` (or one full stream) with the default 60 s
deadline. The Python sync client, async client and the JS client gave the same
times to 0.1 s.

| Failure in front of a healthy provider | 0.4.0 first call | 0.4.0 next call | **0.5.0 first call** | 0.5.0 next call |
|---|---|---|---|---|
| Rate limit (429) | 0.3 s | 0.3 s | **0.3 s** | 0.3 s |
| Server error (500) | 0.3 s | 0.3 s | **0.3 s** | 0.3 s |
| Connection refused | 0.3 s | 0.3 s | **0.3 s** | 0.3 s |
| Provider accepts, never answers | failed after 60 s | 0.3 s | **6.3 s** | 0.3 s |
| Unroutable host | failed after 60 s | 0.3 s | **6.3 s** | 0.3 s |
| Slow provider (answers after 8 s) | 8.0 s | 8.0 s | **6.3 s** | 0.3 s |
| Stream stalls before its first token | failed after 60 s | 0.3 s | **3.3 s** | 0.3 s |

What changed in 0.5.0:

- **Hedging.** An attempt still running after a few seconds (3x that key's usual
  latency, clamped: 1.5–6 s for a stream's first token, 4–12 s for a whole
  answer; 3 s / 6 s when unmeasured) gets a parallel attempt on the next
  provider, and the first answer wins. That is the 6.3 s and 3.3 s above.
- **Smart routing** ranks providers by measured latency, so the second call
  goes straight to the healthy provider (0.4.0 kept trying the slow one first).
- **A 10 s connect timeout** (Python; Node's fetch already had one): with
  hedging off, the unroutable host fails over at 10.3 s instead of 60 s.

With `--no-hedge` on 0.5.0 a hung provider still costs the full 60 s deadline —
that is what hedging is for.

## Reading the numbers honestly

- This isolates freelm's own decision-making. Real providers add their own
  latency (typically 0.3–3 s per answer on free tiers), so real calls are slower
  by that much; the *shape* — one round trip per error, a few seconds per hang,
  near-zero on the next call — is what carries over.
- The hedge delay adapts per key: a provider that usually answers in 300 ms is
  raced after the floor (4 s for answers, 1.5 s for streams), a slow one later.
- A hedge spends one extra request on the slow path only. Calls that answer
  normally are never duplicated.
