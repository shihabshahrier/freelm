"""OpenAI-shaped response objects for the compat shim (attribute access + dict)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ._types import ChatResponse


@dataclass
class CompatMessage:
    role: str
    content: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def model_dump(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class CompatChoice:
    index: int
    message: CompatMessage
    finish_reason: Optional[str]

    def model_dump(self) -> Dict[str, Any]:
        return {"index": self.index, "message": self.message.model_dump(), "finish_reason": self.finish_reason}


@dataclass
class CompatUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def model_dump(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class CompatCompletion:
    id: Optional[str]
    model: Optional[str]
    provider: Optional[str]
    choices: List[CompatChoice]
    usage: CompatUsage
    object: str = "chat.completion"

    def model_dump(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "model": self.model,
            "provider": self.provider,
            "choices": [c.model_dump() for c in self.choices],
            "usage": self.usage.model_dump(),
        }


def wrap_completion(resp: ChatResponse) -> CompatCompletion:
    choices = [
        CompatChoice(
            index=c.index,
            message=CompatMessage(role=c.message.role, content=c.message.content, tool_calls=c.message.tool_calls),
            finish_reason=c.finish_reason,
        )
        for c in resp.choices
    ]
    u = resp.usage
    return CompatCompletion(
        id=resp.id,
        model=resp.model,
        provider=resp.provider,
        choices=choices,
        usage=CompatUsage(u.prompt_tokens, u.completion_tokens, u.total_tokens),
    )
