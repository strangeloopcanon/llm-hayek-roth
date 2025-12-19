from __future__ import annotations

import json

from econ_llm_preferences_experiment.env import OpenAIEnv
from econ_llm_preferences_experiment.openai_client import (
    OpenAIClient,
    _extract_text,
    _extract_usage,
    _hash_payload,
)


def test_extract_text_and_usage() -> None:
    raw = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "hello"}]},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    assert _extract_text(raw) == "hello"
    usage = _extract_usage(raw)
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5


def test_responses_create_uses_cache(tmp_path) -> None:
    env = OpenAIEnv(api_key="sk-test", base_url="https://example.com/v1", model="gpt-5.2")
    client = OpenAIClient(env=env, cache_dir=tmp_path, max_calls=0)

    body = {
        "model": env.model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 800,
    }
    key = _hash_payload(body)
    cached = {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "cached"}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    (tmp_path / f"{key}.json").write_text(json.dumps(cached), encoding="utf-8")

    resp = client.responses_create(input_text="hi", use_cache=True)
    assert resp.text == "cached"
