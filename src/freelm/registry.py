"""Virtual-model registry: map aliases like ``auto`` / ``chat:fast`` to concrete ids.

Model availability on free tiers changes often, so this is intentionally a plain
data table you can edit, plus a resolver. Pass ``models=[...]`` to a provider to
override entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class ModelSpec:
    id: str
    tags: Tuple[str, ...] = field(default_factory=tuple)
    ctx: int = 0
    free: bool = True


# size keywords that map onto tags
_SIZE_ALIASES = {"best": "large", "big": "large", "mini": "small", "cheap": "small", "lite": "small"}
_VIRTUAL = {"auto", "chat", "default", "large", "fast", "small"} | set(_SIZE_ALIASES)


def resolve_models(models: List[ModelSpec], alias: str) -> List[str]:
    """Return an ordered list of concrete model ids for ``alias``.

    - exact concrete id -> just that id
    - ``auto`` / ``chat`` / ``default`` -> all chat models (registry order)
    - ``<size>`` or ``chat:<size>`` -> models tagged with that size, else all chat
    - anything else -> treated as a concrete passthrough id
    """
    ids = [m.id for m in models]
    if alias in ids:
        return [alias]

    a = alias.strip().lower()
    base, _, size = a.partition(":")

    if base not in _VIRTUAL:
        return [alias]  # unknown -> a concrete model id (possibly with a suffix like ":free")

    want = size or (base if base not in {"auto", "chat", "default"} else "")
    want = _SIZE_ALIASES.get(want, want)

    if want in {"large", "fast", "small"}:
        sized = [m.id for m in models if want in m.tags]
        if sized:
            return sized

    chat = [m.id for m in models if "chat" in m.tags]
    return chat or ids
