"""Tiny TTL disk cache for discovered model lists.

Mirrors the openrouter-free skill: JSON file, TTL (default 1h, env override),
0o600 perms. Path: $FREELM_CACHE_DIR or ~/.cache/freelm/models-<provider>.json
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, List, Optional

DEFAULT_TTL = 3600.0


def cache_dir() -> str:
    return os.environ.get("FREELM_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "freelm")


def default_ttl() -> float:
    raw = os.environ.get("FREELM_CACHE_TTL")
    if not raw:
        return DEFAULT_TTL
    try:
        ttl = float(raw)
    except ValueError:
        return DEFAULT_TTL
    return ttl if 0 <= ttl < float("inf") else DEFAULT_TTL


def _path(name: str) -> str:
    safe = name.replace("/", "_")
    return os.path.join(cache_dir(), f"models-{safe}.json")


def load(name: str) -> Optional[List[Any]]:
    try:
        with open(_path(name), "r", encoding="utf-8") as f:
            entry = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict):
        return None  # foreign/corrupt file: ignore it, a live fetch will rewrite it
    try:
        if time.time() > float(entry.get("expires_at") or 0):
            return None
    except (TypeError, ValueError):
        return None
    data = entry.get("data")
    if not isinstance(data, list):
        return None
    return [m for m in data if isinstance(m, dict)]


def save(name: str, data: List[Any], ttl: Optional[float] = None) -> None:
    p = _path(name)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        entry = {"data": data, "expires_at": time.time() + (ttl if ttl is not None else default_ttl())}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(entry, f)
        os.chmod(p, 0o600)
    except OSError:
        pass  # cache is best-effort; never fatal


def clear(name: str) -> None:
    try:
        os.remove(_path(name))
    except OSError:
        pass
