"""freelm CLI — chat / models / health / doctor / serve from the terminal.

    freelm chat "explain failover in one line" [--model auto] [--stream]
    freelm models [--provider openrouter]
    freelm health
    freelm doctor [--json]                    # live-check every key, with fixes
    freelm serve [--port 4000] [--api-key K]  # local OpenAI-compatible endpoint
    freelm --version

Keys come from the environment (same vars as the library; see .env.example).
stdlib-only on purpose: argparse + the library itself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from ._version import __version__
from .errors import (
    AuthError,
    BadRequest,
    ConfigError,
    FreeLLMError,
    ModelNotFound,
    NoProvidersAvailable,
    QuotaExhausted,
    RateLimited,
    Transient,
)
from .strategy import STRATEGIES


def _keyless() -> str:
    """The CLI is for trying things: with no keys it falls back to the keyless
    endpoints (with a notice) unless FREELM_KEYLESS says otherwise."""
    return os.getenv("FREELM_KEYLESS") or "auto"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="freelm", description="Free, always-up LLM client over free-tier providers.")
    p.add_argument("--version", action="version", version=f"freelm {__version__}")
    sub = p.add_subparsers(dest="command")

    c = sub.add_parser("chat", help="send a prompt and print the reply")
    c.add_argument("prompt", help="the user prompt")
    c.add_argument("--model", default="auto", help="virtual alias or concrete model id (default: auto)")
    c.add_argument("--strategy", default="smart", choices=STRATEGIES, help="provider ranking strategy")
    c.add_argument("--stream", action="store_true", help="stream tokens as they arrive")

    m = sub.add_parser("models", help="list available (discovered) models per provider")
    m.add_argument("--provider", default=None, help="only this provider (e.g. openrouter)")

    sub.add_parser("health", help="per-key state of this process (no network; see `doctor`)")

    d = sub.add_parser("doctor", help="live-check every configured key and say how to fix failures")
    d.add_argument("--json", action="store_true", help="machine-readable output")
    d.add_argument("--timeout", type=float, default=30.0, help="seconds per key (default: 30)")

    s = sub.add_parser("serve", help="run a local OpenAI-compatible endpoint (/v1/chat/completions, /v1/models)")
    s.add_argument("--host", default=os.getenv("FREELM_HOST", "127.0.0.1"), help="bind address (default: 127.0.0.1)")
    s.add_argument("--port", type=int, default=int(os.getenv("FREELM_PORT") or "4000"), help="port (default: 4000)")
    s.add_argument("--api-key", default=os.getenv("FREELM_SERVER_KEY"),
                   help="require this bearer token from clients (env FREELM_SERVER_KEY)")
    s.add_argument("--strategy", default="smart", choices=STRATEGIES, help="provider ranking strategy")
    s.add_argument("--cors", action="store_true", help="allow browser apps on other origins")
    s.add_argument("--no-fallback", action="store_true",
                   help="don't serve unknown model ids (e.g. a tool's default 'gpt-4o') with 'auto'")
    s.add_argument("--quiet", action="store_true", help="no per-request log lines")
    return p


def _cmd_chat(args: argparse.Namespace) -> int:
    from .client import FreeLLM

    with FreeLLM.from_env(strategy=args.strategy, keyless=_keyless()) as llm:
        if args.stream:
            for chunk in llm.stream(args.prompt, model=args.model):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            r = llm.chat(args.prompt, model=args.model)
            print(r.text)
            print(f"[{r.provider}/{r.model}]", file=sys.stderr)
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from .config import providers_from_env
    from .discovery import discover_sync

    provs = providers_from_env(_keyless())
    if args.provider:
        provs = [p for p in provs if p.name == args.provider]
        if not provs:
            print(f"no provider named {args.provider!r} configured", file=sys.stderr)
            return 2
    for p in provs:
        if p.discover and not p._discovered:
            discover_sync(p)
        print(f"{p.name}:")
        for m in p.models:
            tags = ",".join(m.tags)
            ctx = f" ctx={m.ctx}" if m.ctx else ""
            print(f"  {m.id}  [{tags}]{ctx}")
    return 0


def _cmd_health(_args: argparse.Namespace) -> int:
    from .client import FreeLLM

    with FreeLLM.from_env(keyless=_keyless()) as llm:
        for row in llm.health():
            print(
                f"{row['provider']:11} {row['key']:18} ready={str(row['ready']):5} "
                f"breaker={row['breaker']:9} rpd={row['rpd_used']}/{row['rpd'] or '-'} "
                f"last_error={row['last_error']}"
            )
    print("(state of this process only — run `freelm doctor` to test the keys live)", file=sys.stderr)
    return 0


# -- doctor ---------------------------------------------------------------------


def _short(exc: BaseException) -> str:
    """The human part of a provider error body (JSON ``message``/``detail``)."""
    msg = getattr(exc, "message", "") or str(exc)
    try:
        obj: Any = json.loads(msg)
        if isinstance(obj, list) and obj:
            obj = obj[0]
        if isinstance(obj, dict) and isinstance(obj.get("errors"), list) and obj["errors"]:
            obj = obj["errors"][0]  # Cloudflare: {"errors": [{"code": ..., "message": ...}]}
        if isinstance(obj, dict):
            err = obj.get("error", obj)
            if isinstance(err, dict):
                msg = err.get("message") or err.get("detail") or err.get("title") or msg
            elif isinstance(err, str):
                msg = err
    except ValueError:
        pass
    return " ".join(str(msg).split())[:110]


def _diagnose(exc: Optional[BaseException], signup: str) -> Tuple[str, str, bool]:
    """(label, detail, key_works) for the last error of a failed check."""
    if exc is None:
        return "FAIL", "no model could be tried (all retired or benched)", False
    code = getattr(exc, "status", "") or ""
    s = _short(exc)
    if isinstance(exc, AuthError):
        return "FAIL", f"key rejected ({code}): {s} — get a new free key: {signup}", False
    if isinstance(exc, QuotaExhausted):
        return "FAIL", f"no free quota/credits left ({code}): {s}", False
    if isinstance(exc, RateLimited):
        return "WARN", f"key works but is rate-limited right now ({code})", True
    if isinstance(exc, ModelNotFound):
        return "FAIL", f"no usable model ({code}): {s}", False
    if isinstance(exc, Transient):
        return "WARN", f"provider unreachable or overloaded ({code or 'network'}): {s}", False
    if isinstance(exc, BadRequest):
        return "FAIL", f"request rejected ({code}): {s}", False
    return "FAIL", s, False


def _check(p: Any, signup: str, timeout: float) -> Dict[str, Any]:
    """One tiny real request through ``p`` (a one-provider client)."""
    from .client import FreeLLM

    row: Dict[str, Any] = {"provider": p.name, "key": p.keys[0].masked(), "model": None, "latency_ms": None}
    llm = FreeLLM([p], max_attempts=4, timeout=timeout, persist=False)  # test the key, not saved state
    try:
        r = llm.chat("Reply with the single word: ok", max_tokens=16, temperature=0)
        row.update(status="OK", works=True, model=r.model, latency_ms=round(r.latency_ms),
                   detail=f"{r.model} · {r.latency_ms:.0f} ms")
    except NoProvidersAvailable as e:
        label, detail, works = _diagnose(e.attempts[-1][1] if e.attempts else None, signup)
        row.update(status=label, works=works, detail=detail)
    except FreeLLMError as e:
        label, detail, works = _diagnose(e, signup)
        row.update(status=label, works=works, detail=detail)
    finally:
        llm.close()
    return row


def _cmd_doctor(args: argparse.Namespace) -> int:
    from ._keys import mask_key
    from .config import KEYLESS, PROVIDER_ENV, build_provider, env_keys, env_vars, keyless_mode

    configured = [(spec, env_keys(spec)) for spec in PROVIDER_ENV]
    missing = [spec for spec, keys in configured if not keys]
    have_keys = any(keys for _, keys in configured)
    check_keyless = keyless_mode(_keyless()) != "never"
    missing_json = [{"provider": s.name, "env": s.key_vars[0], "signup": s.signup_url} for s in missing]

    if not args.json:
        n = sum(len(k) for _, k in configured)
        if have_keys:
            print(f"freelm {__version__} doctor — {n} key(s), one tiny request each\n")
        else:
            print("no provider keys found in the environment. Free keys (no credit card needed for most):\n")
            for s in missing:
                print(f"  {s.name:11} export {' '.join(v + '=...' for v in env_vars(s))}   {s.signup_url}")
            print("\nSet one or more, then run `freelm doctor` again.")
    rows: List[Dict[str, Any]] = []
    for spec, keys in configured:
        for key in keys:
            try:
                p = build_provider(spec, key)
            except ConfigError as e:  # e.g. a Cloudflare token without its account id
                row = {"provider": spec.name, "key": mask_key(key), "model": None, "latency_ms": None,
                       "status": "FAIL", "works": False, "detail": str(e)}
            else:
                row = _check(p, spec.signup_url, args.timeout)
            rows.append(row)
            if not args.json:
                print(f"  {row['provider']:11} {row['key']:16} {row['status']:4}  {row['detail']}", flush=True)

    keyless_rows: List[Dict[str, Any]] = []
    if check_keyless:
        if not args.json:
            print("\nkeyless endpoints (no signup; low limits; free routes may log prompts):")
        for cls in KEYLESS:
            if have_keys and any(p == cls.name for p in (s.name for s, k in configured if k)):
                continue  # already checked with your key
            row = _check(cls(), "", args.timeout)
            keyless_rows.append(row)
            if not args.json:
                print(f"  {row['provider']:11} {row['key']:16} {row['status']:4}  {row['detail']}", flush=True)

    working = [r for r in rows if r["works"]]
    keyless_ok = [r for r in keyless_rows if r["works"]]
    if args.json:
        print(json.dumps({"keys": rows, "keyless": keyless_rows, "missing": missing_json}, indent=2))
    else:
        if have_keys and missing:
            print("\nnot configured (more free capacity, more failover):")
            for s in missing:
                print(f"  {s.name:11} {' + '.join(env_vars(s)):22} {s.signup_url}")
        ready = sorted({r["provider"] for r in working + keyless_ok})
        parts = [f"{len(working)} of {len(rows)} key(s) working" if rows else "no keys configured"]
        if keyless_rows:
            parts.append(f"{len(keyless_ok)} keyless endpoint(s) up")
        print("\n" + ", ".join(parts) + (f" — ready: {', '.join(ready)}" if ready else ""))
    if working or keyless_ok:
        return 0
    return 1 if have_keys else 2


def _cmd_serve(args: argparse.Namespace) -> int:
    from .config import providers_from_env
    from .server import serve

    serve(
        providers_from_env(_keyless()),
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        cors=args.cors,
        fallback_model=None if args.no_fallback else "auto",
        quiet=args.quiet,
        strategy=args.strategy,
    )
    return 0


_COMMANDS = {
    "chat": _cmd_chat,
    "models": _cmd_models,
    "health": _cmd_health,
    "doctor": _cmd_doctor,
    "serve": _cmd_serve,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        return _COMMANDS[args.command](args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except FreeLLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except OSError as e:  # e.g. serve: port already in use
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
