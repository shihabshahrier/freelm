"""freelm CLI — chat / models / health from the terminal.

    freelm chat "explain failover in one line" [--model auto] [--stream]
    freelm models [--provider openrouter]
    freelm health
    freelm --version

Keys come from the environment (same vars as the library; see .env.example).
stdlib-only on purpose: argparse + the library itself.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from ._version import __version__
from .errors import ConfigError, FreeLLMError
from .strategy import STRATEGIES


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="freelm", description="Free, always-up LLM client over free-tier providers.")
    p.add_argument("--version", action="version", version=f"freelm {__version__}")
    sub = p.add_subparsers(dest="command")

    c = sub.add_parser("chat", help="send a prompt and print the reply")
    c.add_argument("prompt", help="the user prompt")
    c.add_argument("--model", default="auto", help="virtual alias or concrete model id (default: auto)")
    c.add_argument("--strategy", default="priority", choices=STRATEGIES, help="provider ranking strategy")
    c.add_argument("--stream", action="store_true", help="stream tokens as they arrive")

    m = sub.add_parser("models", help="list available (discovered) models per provider")
    m.add_argument("--provider", default=None, help="only this provider (e.g. openrouter)")

    sub.add_parser("health", help="per-key health: readiness, breaker, quota, last error")
    return p


def _cmd_chat(args: argparse.Namespace) -> int:
    from .client import FreeLLM

    with FreeLLM.from_env(strategy=args.strategy) as llm:
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

    provs = providers_from_env()
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

    with FreeLLM.from_env() as llm:
        for row in llm.health():
            print(
                f"{row['provider']:11} {row['key']:18} ready={str(row['ready']):5} "
                f"breaker={row['breaker']:9} rpd={row['rpd_used']}/{row['rpd'] or '-'} "
                f"last_error={row['last_error']}"
            )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        if args.command == "chat":
            return _cmd_chat(args)
        if args.command == "models":
            return _cmd_models(args)
        return _cmd_health(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except FreeLLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
