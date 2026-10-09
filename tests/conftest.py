from typing import Any, Dict

import pytest


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    # Never read or write the developer's real ~/.cache/freelm from unit tests:
    # a cached live model list would silently change what the router picks.
    monkeypatch.setenv("FREELM_CACHE_DIR", str(tmp_path / "freelm-cache"))
    monkeypatch.delenv("FREELM_PERSIST", raising=False)


def ok_payload(content: str = "hello", model: str = "test-model") -> Dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
