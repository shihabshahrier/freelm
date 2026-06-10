"""Virtual-model registry: map aliases like ``auto`` / ``chat:fast`` to concrete ids.

Model availability on free tiers changes often, so this is intentionally a plain
data table you can edit, plus a resolver. Pass ``models=[...]`` to a provider to
override entirely; give specs a ``priority`` to control order without replacing
the list.
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
    priority: int = 0  # lower = preferred (stable: ties keep list order)


# size keywords that map onto tags
_SIZE_ALIASES = {"best": "large", "big": "large", "mini": "small", "cheap": "small", "lite": "small"}
# tags that can be asked for directly: `chat:tools`, `vision`, `reasoning`, ...
_TAG_ALIASES = {"large", "fast", "small", "tools", "vision", "reasoning"}
_VIRTUAL = {"auto", "chat", "default"} | _TAG_ALIASES | set(_SIZE_ALIASES)


def resolve_models(models: List[ModelSpec], alias: str) -> List[str]:
    """Return an ordered list of concrete model ids for ``alias``.

    - exact concrete id -> just that id
    - ``auto`` / ``chat`` / ``default`` -> all chat models (priority, then list order)
    - ``<tag>`` or ``chat:<tag>`` (size/tools/vision/reasoning) -> models tagged
      with it, else all chat
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

    ordered = sorted(models, key=lambda m: m.priority)  # stable: priority, then list order

    if want in _TAG_ALIASES:
        tagged = [m.id for m in ordered if want in m.tags]
        if tagged:
            return tagged

    chat = [m.id for m in ordered if "chat" in m.tags]
    return chat or [m.id for m in ordered]
