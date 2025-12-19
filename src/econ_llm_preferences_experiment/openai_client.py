from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from econ_llm_preferences_experiment.env import OpenAIEnv, get_openai_env
from econ_llm_preferences_experiment.logging_utils import get_logger, log

logger = get_logger(__name__)


class OpenAIError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAIUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class OpenAIResponse:
    text: str
    raw: Mapping[str, Any]
    usage: OpenAIUsage


def _hash_payload(payload: Mapping[str, Any]) -> str:
    packed = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def _extract_text(resp: Mapping[str, Any]) -> str:
    """
    Extracts the first text segment from the Responses API output.
    """
    output = resp.get("output", [])
    for item in output:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") in {"output_text", "text"}:
                return str(part.get("text", ""))
    return ""


def _extract_usage(resp: Mapping[str, Any]) -> OpenAIUsage:
    usage = resp.get("usage", {})
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    return OpenAIUsage(
        input_tokens=int(input_tokens) if isinstance(input_tokens, (int, float)) else None,
        output_tokens=int(output_tokens) if isinstance(output_tokens, (int, float)) else None,
    )


class OpenAIClient:
    def __init__(
        self,
        env: OpenAIEnv | None = None,
        *,
        cache_dir: str | os.PathLike[str] = ".cache/llm",
        max_calls: int = 9,
    ) -> None:
        self.env = env or get_openai_env()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls
        self._calls_made = 0

    def responses_create(
        self,
        *,
        input_text: str,
        model: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_output_tokens: int = 800,
        use_cache: bool = True,
        timeout_s: float = 60.0,
    ) -> OpenAIResponse:
        if not self.env.api_key:
            raise OpenAIError("OPENAI_API_KEY is missing/empty (check .env).")

        body: dict[str, Any] = {
            "model": model or self.env.model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": input_text}],
                }
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
        }

        cache_key = _hash_payload(body)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if use_cache and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return OpenAIResponse(
                text=_extract_text(cached), raw=cached, usage=_extract_usage(cached)
            )

        if self._calls_made >= self.max_calls:
            raise OpenAIError(f"LLM call budget exceeded (max_calls={self.max_calls}).")

        self._calls_made += 1
        url = f"{self.env.base_url.rstrip('/')}/responses"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.env.api_key}",
                "Content-Type": "application/json",
            },
        )

        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                log(
                    logger,
                    level=20,
                    message="openai_request",
                    attempt=attempt,
                    cached=False,
                    model=body["model"],
                )
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # nosec B310
                    raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
                cache_path.write_text(
                    json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                return OpenAIResponse(
                    text=_extract_text(parsed), raw=parsed, usage=_extract_usage(parsed)
                )
            except urllib.error.HTTPError as e:
                last_err = e
                status = getattr(e, "code", None)
                is_transient = status in {429, 500, 502, 503, 504}
                if not is_transient:
                    raise OpenAIError(f"OpenAI HTTP error: {status}") from e
            except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as e:
                last_err = e

            sleep_s = 0.5 * (2 ** (attempt - 1)) + random.random() * 0.25  # nosec B311
            time.sleep(sleep_s)

        raise OpenAIError("OpenAI request failed after retries.") from last_err
