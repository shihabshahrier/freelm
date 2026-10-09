# Security policy

freelm handles your provider API keys, so we take key exposure seriously.

## Reporting a vulnerability

Please **don't open a public issue**. Use GitHub's private
[security advisory form](https://github.com/shihabshahrier/freelm/security/advisories/new)
or email shahriarlabs@gmail.com. You'll get an acknowledgement within a few
days; fixes ship as patch releases for both the PyPI and npm packages.

## What freelm guarantees

- Keys are read from the environment or passed in code; freelm never writes a
  raw key to disk, logs, events, `repr()`/`inspect` output or error messages
  (masked as `sk-or-...abcd`). Persisted state stores only a 12-char SHA-256
  prefix of each key.
- Cache and state files are created `0600`.
- `freelm serve` binds to `127.0.0.1` by default. If you expose it beyond
  localhost, set `--api-key` (or `FREELM_SERVER_KEY`) — otherwise anyone who
  can reach the port can spend your free quota.
- Requests go only to the provider endpoints you configure; freelm itself has
  no telemetry.

## Supported versions

Only the latest release of each package receives fixes.
