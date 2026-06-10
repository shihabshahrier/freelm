"""Provider-agnostic data types (OpenAI-shaped, but pure dataclasses)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class Message:
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

    @classmethod
    def from_any(cls, m: Union["Message", Dict[str, Any], str]) -> "Message":
        if isinstance(m, Message):
            return m
        if isinstance(m, str):
            return cls(role="user", content=m)
        if isinstance(m, dict):
            return cls(
                role=m.get("role", "user"),
                content=m.get("content"),
                name=m.get("name"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
        raise TypeError(f"unsupported message type: {type(m)!r}")


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Usage":
        d = d or {}
        return cls(
            prompt_tokens=int(d.get("prompt_tokens") or 0),
            completion_tokens=int(d.get("completion_tokens") or 0),
            total_tokens=int(d.get("total_tokens") or 0),
        )


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: Optional[str] = None


@dataclass
class ChatResponse:
    id: Optional[str]
    model: Optional[str]
    provider: Optional[str]
    choices: List[Choice]
    usage: Usage
    latency_ms: float = 0.0
    raw: Optional[Dict[str, Any]] = None

    @property
    def text(self) -> str:
        if not self.choices:
            return ""
        return self.choices[0].message.content or ""

    @property
    def tool_calls(self) -> Optional[List[Dict[str, Any]]]:
        """Tool calls of the first choice, if the model requested any."""
        if not self.choices:
            return None
        return self.choices[0].message.tool_calls

    def __str__(self) -> str:  # so print(resp) gives the text
        return self.text


@dataclass
class Event:
    """One observability event emitted via ``FreeLLM(on_event=...)``.

    ``kind`` is ``"attempt" | "success" | "error" | "wait" | "discovery"``.
    ``key`` is always masked — never a raw API key.
    """

    kind: str
    provider: Optional[str] = None
    key: Optional[str] = None
    model: Optional[str] = None
    status: Optional[int] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    attempt: int = 0


_SAMPLING_FIELDS = ("temperature", "max_tokens", "top_p", "stop", "seed", "frequency_penalty", "presence_penalty")


@dataclass
class ChatRequest:
    messages: List[Dict[str, Any]]
    model: Union[str, Tuple[str, ...]] = "auto"   # alias, or ordered fallback chain
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def payload(self, concrete_model: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": concrete_model, "messages": self.messages}
        for k in _SAMPLING_FIELDS:
            v = getattr(self, k)
            if v is not None:
                body[k] = v
        body.update(self.extra)
        return body


def build_request(messages: Any, model: Union[str, Sequence[str]], kw: Dict[str, Any]) -> ChatRequest:
    if not isinstance(messages, (list, tuple)):
        messages = [messages]
    msgs = [Message.from_any(m).to_dict() for m in messages]
    fields = {k: kw.pop(k) for k in list(kw) if k in _SAMPLING_FIELDS}
    m = model if isinstance(model, str) else tuple(model)
    return ChatRequest(messages=msgs, model=m, extra=dict(kw), **fields)
