"""Opt-in persistent key state (rpd counters, cooldowns, disabled flags).

Survives process restarts so a fresh run doesn't re-burn keys that are already
exhausted or dead. JSON file in the cache dir, 0600, atomic replace. The schema
is shared with the JS package (``js/src/state.ts``) so both can read it:

    {"<provider>:<sha256(key)[:12]>": {"rpd_used": int, "rpd_reset_wall": float,
     "cooldown_until_wall": float, "disabled": bool, "last_error": str|null}}

Raw keys are never written — only a short hash. Wall-clock timestamps in the
file are converted to/from the in-process monotonic clock on load/save.
Multi-process use is last-writer-wins (best effort, documented).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

from ._cache import cache_dir


def _key_id(provider_name: str, key: str) -> str:
    return f"{provider_name}:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"


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
                k.rpd_used = int(e.get("rpd_used") or 0)
                rr = float(e.get("rpd_reset_wall") or 0)
                if rr > wall:
                    k.rpd_reset = now_mono + (rr - wall)
                cu = float(e.get("cooldown_until_wall") or 0)
                if cu > wall:
                    k.cooldown_until = now_mono + (cu - wall)
                k.disabled = bool(e.get("disabled", False))
                if e.get("last_error"):
                    k.last_error = e["last_error"]

    def save(self, providers: List[Any], now_mono: float) -> None:
        # merge over existing entries so other processes/providers aren't clobbered
        data = self._read()
        wall = time.time()
        for p in providers:
            for k in p.keys:
                data[_key_id(p.name, k.key)] = {
                    "rpd_used": k.rpd_used,
                    "rpd_reset_wall": wall + (k.rpd_reset - now_mono) if k.rpd_reset > 0 else 0,
                    "cooldown_until_wall": wall + (k.cooldown_until - now_mono) if k.cooldown_until > now_mono else 0,
                    "disabled": k.disabled,
                    "last_error": k.last_error,
                }
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except OSError:
            pass  # persistence is best-effort; never fatal
