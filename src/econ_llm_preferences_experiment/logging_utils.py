from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload.update(context)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        if isinstance(handler.formatter, JsonlFormatter):
            return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonlFormatter())
    logger.handlers = [handler]
    logger.propagate = False
    return logger


def log(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    logger.log(level, message, extra={"context": fields})
