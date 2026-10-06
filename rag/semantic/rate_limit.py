"""Evaluation-side provider pacing and 429 handling (eval only).

Free-tier reality: ~15 generate_content requests per rolling minute. The
normal interactive app path is untouched — this limiter wraps provider
callables inside the evaluation runner only.

Two mechanisms (both required, either alone is insufficient):

1. Question-level pacing: the runner waits ~3s between consecutive cases.
2. Rolling request-window guard: request timestamps within the trailing
   60 seconds must stay below a conservative ceiling (default 14), while a
   small burst inside one question is permitted when capacity exists.

429 / quota exhaustion is an INFRASTRUCTURE event, never an accuracy
failure. Bounded infra retry (default 2 per provider call) honors
Retry-After + buffer; exhausted rows are recorded EVAL_INFRA_ERROR and
excluded from the accuracy denominator. Semantic/model failures
(malformed output, validation errors, provider non-quota errors) are
never retried here.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MAX_REQUESTS = 14
DEFAULT_WINDOW_SECONDS = 60.0
DEFAULT_MIN_QUESTION_GAP_SECONDS = 3.0
DEFAULT_MAX_INFRA_RETRIES = 2
DEFAULT_RETRY_BUFFER_SECONDS = 5.0
DEFAULT_FALLBACK_RETRY_AFTER_SECONDS = 30.0

_RETRY_AFTER_RE = re.compile(r"retry in\s*([\d.]+)\s*s", re.IGNORECASE)


def is_rate_limit_error(error: Exception) -> bool:
    """Return True only for transport-level quota/rate-limit failures."""
    if getattr(error, "code", None) == 429:
        return True
    message = str(error)
    lowered = message.lower()
    if "429" in message and ("quota" in lowered or "rate" in lowered):
        return True
    if "quota exceeded" in lowered or "rate limit" in lowered:
        return True
    return False


def parse_retry_after_seconds(error: Exception) -> float | None:
    """Extract Retry-After seconds from provider diagnostics, if present."""
    match = _RETRY_AFTER_RE.search(str(error))
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except (TypeError, ValueError):
        return None
    if value < 0 or value > 600:
        return None
    return value


class RateLimiter:
    """Rolling-window request guard with injectable clock/sleeper (testable)."""

    def __init__(
        self,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._clock = clock
        self._sleeper = sleeper
        self._stamps: list[float] = []

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        self._stamps = [stamp for stamp in self._stamps if stamp > cutoff]

    def wait_for_slot(self) -> float:
        """Block until the rolling window has capacity; return waited ms."""
        waited_ms = 0.0
        while True:
            now = self._clock()
            self._prune(now)
            if len(self._stamps) < self._max_requests:
                self._stamps.append(self._clock())
                return waited_ms
            oldest = min(self._stamps)
            delay = max(0.0, oldest + self._window_seconds - now) + 0.1
            self._sleeper(delay)
            waited_ms += delay * 1000.0

    def pending_count(self) -> int:
        """Return current in-window request count (observability)."""
        self._prune(self._clock())
        return len(self._stamps)


class QuestionPacer:
    """Minimum gap between consecutive evaluation cases (fake-clock testable)."""

    def __init__(
        self,
        min_gap_seconds: float = DEFAULT_MIN_QUESTION_GAP_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_gap = min_gap_seconds
        self._clock = clock
        self._sleeper = sleeper
        self._last_end: float | None = None

    def wait_between_questions(self) -> float:
        """Sleep until the minimum inter-question gap elapsed; return slept ms."""
        now = self._clock()
        if self._last_end is None:
            self._last_end = now
            return 0.0
        elapsed = now - self._last_end
        if elapsed >= self._min_gap:
            self._last_end = now
            return 0.0
        delay = self._min_gap - elapsed
        self._sleeper(delay)
        self._last_end = self._clock()
        return delay * 1000.0

    def mark_case_complete(self) -> None:
        self._last_end = self._clock()


@dataclass(slots=True)
class GateCallResult:
    ok: bool
    result: Any = None
    retries: int = 0
    waited_ms: float = 0.0
    error_type: str | None = None
    rate_limited: bool = False


class ProviderGate:
    """Bounded infra-retry wrapper around one provider callable."""

    def __init__(
        self,
        limiter: RateLimiter,
        max_infra_retries: int = DEFAULT_MAX_INFRA_RETRIES,
        retry_buffer_seconds: float = DEFAULT_RETRY_BUFFER_SECONDS,
        fallback_retry_after_seconds: float = DEFAULT_FALLBACK_RETRY_AFTER_SECONDS,
    ) -> None:
        self._limiter = limiter
        self._max_infra_retries = max_infra_retries
        self._retry_buffer = retry_buffer_seconds
        self._fallback_retry_after = fallback_retry_after_seconds

    def call(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> GateCallResult:
        """Run func with pacing + bounded 429 retry. Never retries semantics."""
        waited_ms = self._limiter.wait_for_slot()
        retries = 0
        while True:
            try:
                return GateCallResult(
                    ok=True,
                    result=func(*args, **kwargs),
                    retries=retries,
                    waited_ms=waited_ms,
                )
            except Exception as error:  # noqa: BLE001 - classified below
                if not is_rate_limit_error(error) or retries >= self._max_infra_retries:
                    return GateCallResult(
                        ok=False,
                        retries=retries,
                        waited_ms=waited_ms,
                        error_type=type(error).__name__,
                        rate_limited=is_rate_limit_error(error),
                    )
                retries += 1
                retry_after = parse_retry_after_seconds(error)
                if retry_after is None:
                    retry_after = self._fallback_retry_after
                self._limiter._sleeper(retry_after + self._retry_buffer)
                waited_ms += (retry_after + self._retry_buffer) * 1000.0
                waited_ms += self._limiter.wait_for_slot()


__all__ = [
    "DEFAULT_FALLBACK_RETRY_AFTER_SECONDS",
    "DEFAULT_MAX_INFRA_RETRIES",
    "DEFAULT_MAX_REQUESTS",
    "DEFAULT_MIN_QUESTION_GAP_SECONDS",
    "DEFAULT_RETRY_BUFFER_SECONDS",
    "DEFAULT_WINDOW_SECONDS",
    "GateCallResult",
    "ProviderGate",
    "QuestionPacer",
    "RateLimiter",
    "is_rate_limit_error",
    "parse_retry_after_seconds",
]
