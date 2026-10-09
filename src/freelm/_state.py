"""Opt-in persistent key state (rpd counters, cooldowns, disabled flags).

Survives process restarts so a fresh run doesn't re-burn keys that are already
exhausted or dead. JSON file in the cache dir, 0600, atomic replace. The schema
is shared with the JS package (``js/src/state.ts``) so both can read it:

    {"<provider>:<sha256(key)[:12]>": {"rpd_used": int, "rpd_reset_wall": float,
     "cooldown_until_wall": float, "disabled": bool, "disabled_since_wall": float,
     "last_error": str|null}}

Raw keys are never written — only a short hash. Wall-clock timestamps in the
file are converted to/from the in-process monotonic clock on load/save. A
persisted ``disabled`` flag expires after ``DISABLED_TTL`` (the account may have
been topped up / the key re-enabled), so a dead key gets one fresh try per day.
Multi-process use is last-writer-wins (best effort, documented).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from ._cache import cache_dir

DISABLED_TTL = 86400.0


def _key_id(provider_name: str, key: str) -> str:
    return f"{provider_name}:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"


def _num(v: Any) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


class StateStore:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or os.path.join(cache_dir(), "state.json")

    def _read(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def load_into(self, providers: List[Any], now_mono: float) -> None:
        data = self._read()
        if not data:
            return
        wall = time.time()
        for p in providers:
            for k in p.keys:
                e = data.get(_key_id(p.name, k.key))
                if not isinstance(e, dict):
                    continue
                rr = _num(e.get("rpd_reset_wall"))
                if rr > wall:  # same daily window as when saved -> keep its count
                    k.rpd_used = int(_num(e.get("rpd_used")))
                    k.rpd_reset = now_mono + (rr - wall)
                cu = _num(e.get("cooldown_until_wall"))
                if cu > wall:
                    k.cooldown_until = now_mono + (cu - wall)
                since = _num(e.get("disabled_since_wall")) or wall
                if bool(e.get("disabled", False)) and wall - since < DISABLED_TTL:
                    k.disabled = True
                    k.disabled_since_wall = since
                else:
                    k.disabled_since_wall = 0.0  # expired: if it fails again, that's a fresh disable
                if isinstance(e.get("last_error"), str):
                    k.last_error = e["last_error"]

    def save(self, providers: List[Any], now_mono: float) -> None:
        # merge over existing entries so other processes/providers aren't clobbered
        data = self._read()
        wall = time.time()
        for p in providers:
            for k in p.keys:
                kid = _key_id(p.name, k.key)
                since = 0.0
                if k.disabled:
                    prev = data.get(kid) if isinstance(data.get(kid), dict) else {}
                    inherited = _num(prev.get("disabled_since_wall")) if prev.get("disabled") else 0.0
                    if inherited and wall - inherited >= DISABLED_TTL:
                        inherited = 0.0  # another process' expired disable — this one is new
                    since = getattr(k, "disabled_since_wall", 0.0) or inherited or wall
                    k.disabled_since_wall = since
                data[kid] = {
                    "rpd_used": k.rpd_used,
                    "rpd_reset_wall": wall + (k.rpd_reset - now_mono) if k.rpd_reset > 0 else 0,
                    "cooldown_until_wall": wall + (k.cooldown_until - now_mono) if k.cooldown_until > now_mono else 0,
                    "disabled": k.disabled,
                    "disabled_since_wall": since,
                    "last_error": k.last_error,
                }
        try:
            d = os.path.dirname(self.path)
            os.makedirs(d, exist_ok=True)
            # unique temp file per writer: concurrent processes must never swap
            # each other's half-written file into place
            fd, tmp = tempfile.mkstemp(prefix=".state.", suffix=".tmp", dir=d)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.chmod(tmp, 0o600)
                os.replace(tmp, self.path)
            except BaseException:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
        except OSError:
            pass  # persistence is best-effort; never fatal
