# Contributing to freelm

Thanks for helping keep free LLM access reliable. freelm ships **two packages
from one repo** — Python (`src/freelm/`, PyPI) and TypeScript (`js/src/`, npm) —
and they must behave the same.

## The one rule: parity

Any behavior change lands in **both languages in the same PR, with tests in
both**. The file mapping is 1:1 (`client.py` ↔ `client.ts`, `_engine.py` ↔
`engine.ts`, `server.py` ↔ `server.ts`, ...) — see [AGENTS.md](AGENTS.md).

## Setup

```bash
# Python (>= 3.9)
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/ruff check src tests examples

# TypeScript (Node >= 20)
cd js && npm ci
npm test && npm run typecheck && npm run build
```

Unit tests never touch the network (respx in Python, a stubbed `fetch` in TS).
To try your change against the real free tiers, put keys in `.env` (never
commit it) and run:

```bash
set -a; . ./.env; set +a
freelm doctor                     # which keys work right now
python examples/e2e_smoke.py      # and: cd js && node examples/e2e.mjs
```

## Common contributions

- **A free model was retired / renamed.** Update the provider's
  `DEFAULT_MODELS` (Python) *and* `defaultModels` (TS), with a dated comment
  saying how you verified it. Discovery usually hides this, but the fallback
  list matters when `/models` is unreachable.
- **A provider changed its error format.** Add the real response body as a case
  to the `classify` tests in both languages (`tests/test_hardening.py`,
  `js/test/hardening.test.ts`).
- **A new free provider.** It must offer a genuinely free tier via an official
  API (no reverse-engineered endpoints, no paid-only services). Copy an
  existing provider, add it to `PROVIDER_ENV` in both `config` modules, and
  document its signup link and limits.

## Style

Match the surrounding code; keep runtime dependencies at httpx (Python) and
zero (TS). Bump versions only in release PRs (`src/freelm/_version.py`,
`js/package.json`, `js/src/version.ts` together) with a CHANGELOG entry.
