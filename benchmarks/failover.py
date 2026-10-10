"""Reproducible provider-failure simulation for freelm (Python).

Fake providers run on localhost over real HTTP; freelm is used unmodified with
its defaults (timeout 60 s, hedging on, smart routing). Each scenario puts one
misbehaving provider in front of a healthy one and times two calls on the same
client: the first shows how fast freelm gets past the failure, the second shows
whether it learned from it.

    python benchmarks/failover.py               # this checkout, sync + async client
    python benchmarks/failover.py --no-hedge    # hedging off (smart routing stays on)
    python benchmarks/failover.py --installed   # the freelm installed in this environment,
                                                # e.g. `pip install freelm==0.4.0` for a before/after

Fake provider behaviours (first path segment of its base URL):
  ok     200 after 0.3 s            r429   immediate 429        e500   immediate 500
  hang   accepts, never answers     slow8  200 after 8 s        stall  stream headers, then nothing
  sok    stream, first token after 0.3 s
  dead   nothing listens (connection refused)
  blackhole  unroutable address (the connection never completes)
"""
import argparse
import asyncio
import inspect
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if "--installed" not in sys.argv:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("FREELM_CACHE_DIR", tempfile.mkdtemp(prefix="freelm-bench-"))

from freelm import AsyncFreeLLM, FreeLLM, ModelSpec, NoProvidersAvailable, Provider, __version__  # noqa: E402

OK = json.dumps({"id": "x", "model": "m", "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}).encode()


class FakeProvider(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        kind = self.path.strip("/").split("/")[0]
        try:
            if kind == "hang":
                time.sleep(120)
                return
            if kind in ("stall", "sok"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.flush()
                if kind == "stall":
                    time.sleep(120)
                    return
                time.sleep(0.3)
                self.wfile.write(b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')
                return
            time.sleep({"ok": 0.3, "slow8": 8.0}.get(kind, 0.0))
            status = {"r429": 429, "e500": 500}.get(kind, 200)
            body = OK if status == 200 else b'{"error":{"message":"simulated"}}'
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # freelm cancelled this attempt (a hedge won)


SCENARIOS = [
    ("429, then healthy", ["r429", "ok"], False),
    ("500, then healthy", ["e500", "ok"], False),
    ("connection refused, then healthy", ["dead", "ok"], False),
    ("hung provider, then healthy", ["hang", "ok"], False),
    ("unroutable host, then healthy", ["blackhole", "ok"], False),
    ("slow provider (8 s), then healthy", ["slow8", "ok"], False),
    ("stream stalls before first token, then healthy", ["stall", "sok"], True),
]


def providers(port, kinds):
    def url(kind):
        if kind == "dead":
            return "http://127.0.0.1:9/v1"
        if kind == "blackhole":
            return "http://10.255.255.1/v1"
        return f"http://127.0.0.1:{port}/{kind}/v1"

    return [Provider("k", name=f"{k}{i}", base_url=url(k), models=[ModelSpec("m", ("chat",))], rpm=None)
            for i, k in enumerate(kinds)]


def client_kw(hedge):
    """``hedge=`` only where this freelm has it (0.5+); older versions are sequential."""
    return {"hedge": hedge} if "hedge" in inspect.signature(FreeLLM.__init__).parameters else {}


def fmt(dt, ok):
    return f"{dt:5.1f} s" if ok else f"{dt:5.1f} s FAILED"


def run_sync(port, hedge):
    rows = []
    for label, kinds, stream in SCENARIOS:
        with FreeLLM(providers(port, kinds), persist=False, **client_kw(hedge)) as llm:
            cells = []
            for _ in range(2):
                t0 = time.monotonic()
                try:
                    "".join(llm.stream("hi")) if stream else llm.chat("hi")
                    ok = True
                except NoProvidersAvailable:
                    ok = False
                cells.append(fmt(time.monotonic() - t0, ok))
        rows.append((label, *cells))
        print(f"  sync  {label:48} {cells[0]:>16}   {cells[1]:>16}", flush=True)
    return rows


async def run_async(port, hedge):
    rows = []
    for label, kinds, stream in SCENARIOS:
        async with AsyncFreeLLM(providers(port, kinds), persist=False, **client_kw(hedge)) as llm:
            cells = []
            for _ in range(2):
                t0 = time.monotonic()
                try:
                    if stream:
                        [t async for t in llm.astream("hi")]
                    else:
                        await llm.chat("hi")
                    ok = True
                except NoProvidersAvailable:
                    ok = False
                cells.append(fmt(time.monotonic() - t0, ok))
        rows.append((label, *cells))
        print(f"  async {label:48} {cells[0]:>16}   {cells[1]:>16}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-hedge", action="store_true", help="one attempt at a time (freelm < 0.5 behaviour)")
    ap.add_argument("--sync-only", action="store_true")
    ap.add_argument("--installed", action="store_true", help="benchmark the installed freelm, not this checkout")
    args = ap.parse_args()
    hedge = not args.no_hedge

    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeProvider)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    mode = ("on" if hedge else "off") if client_kw(hedge) else "n/a (sequential)"
    print(f"freelm {__version__} (Python {sys.version.split()[0]}), hedge={mode}, "
          f"defaults otherwise (timeout 60 s)\n")
    print(f"  {'':6}{'scenario':48} {'1st call':>16}   {'2nd call':>16}")
    run_sync(port, hedge)
    if not args.sync_only:
        asyncio.run(run_async(port, hedge))


if __name__ == "__main__":
    main()
