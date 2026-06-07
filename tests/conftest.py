from typing import Any, Dict


def ok_payload(content: str = "hello", model: str = "test-model") -> Dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
