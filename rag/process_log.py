"""Small opt-in process logger for local semantic debugging.

The logger is intentionally stdout-only and disabled by default. It never
records provider prompts, credentials, model chain-of-thought, or database
contents beyond the bounded fields explicitly supplied by callers.
"""

from __future__ import annotations

import json
import os
from typing import Any


PROCESS_LOG_ENV_VAR = "CUCUMBER_PROCESS_LOG"
PROCESS_LOG_PREFIX = "[CUCUMBER_PROCESS]"
MAX_PROCESS_LOG_CHARS = 6000


def process_log_enabled() -> bool:
    raw = os.environ.get(PROCESS_LOG_ENV_VAR, "")
    return isinstance(raw, str) and raw.strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def process_log(stage: str, **fields: Any) -> None:
    """Print one bounded JSON process record when local tracing is enabled."""
    if not process_log_enabled():
        return
    payload = {"stage": str(stage), **fields}
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    if len(rendered) > MAX_PROCESS_LOG_CHARS:
        rendered = json.dumps(
            {
                "stage": str(stage),
                "truncated": True,
                "preview": rendered[: MAX_PROCESS_LOG_CHARS - 120],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    print(f"{PROCESS_LOG_PREFIX} {rendered}", flush=True)


__all__ = [
    "MAX_PROCESS_LOG_CHARS",
    "PROCESS_LOG_ENV_VAR",
    "PROCESS_LOG_PREFIX",
    "process_log",
    "process_log_enabled",
]
