from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(
    path: str | os.PathLike[str] = ".env", *, override: bool = False
) -> Mapping[str, str]:
    """
    Minimal .env loader (KEY=VALUE), purposely avoiding third-party deps.
    """
    env_path = Path(path)
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if key in os.environ and not override:
            loaded[key] = os.environ[key]
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded


@dataclass(frozen=True)
class OpenAIEnv:
    api_key: str
    base_url: str
    model: str


def get_openai_env() -> OpenAIEnv:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    model = os.environ.get("OPENAI_MODEL", "gpt-5.2").strip()
    return OpenAIEnv(api_key=api_key, base_url=base_url, model=model)
