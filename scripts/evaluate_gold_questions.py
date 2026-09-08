"""Execute Gold Questions through the project RAG pipeline and capture results."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # Keep import checks usable in minimal environments.
    def load_dotenv() -> None:
        return None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.answer import answer_question  # noqa: E402
from rag.providers.gemini import make_gemini_callable  # noqa: E402
from rag.qa import ask  # noqa: E402
from rag.router import route_question  # noqa: E402


GOLD_QUESTIONS_PATH = PROJECT_ROOT / "ground_truth" / "rag" / "gold_questions.json"
CURRICULUM_DB_PATH = PROJECT_ROOT / "cucumber_outputs" / "runtime" / "curriculum.db"
TOP_K = 10
ROUTES = {"structured", "semantic", "hybrid"}
DEFAULT_MIN_CALL_INTERVAL = 5.0
MAX_MODEL_RETRIES = 3
FALLBACK_RETRY_DELAY = 30.0


def _exception_text(exc: Exception) -> str:
    values = [str(exc)]
    for name in ("code", "status", "status_code", "reason"):
        value = getattr(exc, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:  # noqa: BLE001 - diagnostic attribute only
                value = None
        if value is not None:
            values.append(str(value))
    return " ".join(values).upper()


def _is_gemini_rate_limit(exc: Exception) -> bool:
    text = _exception_text(exc)
    return bool(re.search(r"\b429\b", text)) and "RESOURCE_EXHAUSTED" in text


def _delay_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    seconds = getattr(value, "seconds", None)
    nanos = getattr(value, "nanos", 0)
    if seconds is not None:
        try:
            return max(0.0, float(seconds) + float(nanos) / 1_000_000_000)
        except (TypeError, ValueError):
            return None
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*", str(value))
    return float(match.group(1)) if match else None


def _retry_delay(exc: Exception) -> float:
    for name in (
        "retry_delay_seconds",
        "retry_after_seconds",
        "retry_delay",
        "retry_after",
    ):
        delay = _delay_value(getattr(exc, name, None))
        if delay is not None:
            return delay

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        for name in ("retry-after", "Retry-After"):
            delay = _delay_value(headers.get(name))
            if delay is not None:
                return delay
    return FALLBACK_RETRY_DELAY


class _SharedGeminiThrottle:
    """Throttle one shared Gemini callable and retry only Gemini 429 exhaustion."""

    def __init__(self, callable_: Any, min_call_interval: float) -> None:
        self._callable = callable_
        self._min_call_interval = min_call_interval
        self._last_call_at: float | None = None
        self.retry_count = 0

    def begin_question(self) -> None:
        self.retry_count = 0

    def _wait_for_interval(self) -> None:
        if self._last_call_at is None:
            return
        remaining = self._min_call_interval - (time.monotonic() - self._last_call_at)
        if remaining > 0:
            time.sleep(remaining)

    def __call__(self, prompt: str) -> str:
        retries = 0
        while True:
            self._wait_for_interval()
            self._last_call_at = time.monotonic()
            try:
                return self._callable(prompt)
            except Exception as exc:  # noqa: BLE001 - retry classification is explicit
                if not _is_gemini_rate_limit(exc) or retries >= MAX_MODEL_RETRIES:
                    raise
                retries += 1
                self.retry_count += 1
                delay = max(
                    _retry_delay(exc),
                    self._min_call_interval
                    - (time.monotonic() - self._last_call_at),
                )
                if delay > 0:
                    time.sleep(delay)


def _json_safe(value: Any) -> Any:
    """Convert pipeline containers, including SQLite row tuples, to JSON data."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _split_pipeline_result(
    route: str,
    result: Any,
) -> tuple[Any, Any]:
    if route == "structured":
        return result, None
    if route == "semantic":
        return None, result
    return result["structured"], result["semantic"]


def _evaluate_question(
    gold: Mapping[str, Any],
    gemini_callable: Any,
) -> dict[str, Any]:
    gemini_callable.begin_question()
    actual_route: str | None = route_question(gold["question"])
    structured_result: Any = None
    semantic_results: Any = None
    final_answer: str | None = None
    error: str | None = None
    execution_success = False

    try:
        response = ask(
            CURRICULUM_DB_PATH,
            gold["question"],
            structured_model_callable=gemini_callable,
            top_k=TOP_K,
        )
        actual_route = response["route"]
        structured_result, semantic_results = _split_pipeline_result(
            actual_route,
            response["result"],
        )
        final_answer = answer_question(
            gold["question"],
            actual_route,
            structured_result=structured_result,
            semantic_chunks=semantic_results,
            answer_model_callable=gemini_callable,
        )
        execution_success = True
    except Exception as exc:  # noqa: BLE001 - capture per-question failures
        error = f"{type(exc).__name__}: {exc}"

    gold_type = gold["type"]
    route_match = (
        actual_route == gold_type if gold_type in ROUTES and actual_route else None
    )
    return {
        "id": gold["id"],
        "type": gold_type,
        "question": gold["question"],
        "expected": gold["expected"],
        "actual_route": actual_route,
        "route_match": route_match,
        "structured_result": _json_safe(structured_result),
        "semantic_results": _json_safe(semantic_results),
        "final_answer": final_answer,
        "execution_success": execution_success,
        "error": error,
        "top_k": TOP_K,
        "model_retry_count": gemini_callable.retry_count,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gold Questions through the actual curriculum RAG pipeline."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="run only the first N Gold Questions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_result.json"),
        metavar="PATH",
        help="output JSON path (default: eval_result.json)",
    )
    parser.add_argument(
        "--min-call-interval",
        type=float,
        default=DEFAULT_MIN_CALL_INTERVAL,
        metavar="SECONDS",
        help="minimum interval between Gemini calls (default: 5)",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if not math.isfinite(args.min_call_interval) or args.min_call_interval < 0:
        parser.error("--min-call-interval must be a finite non-negative number")
    return args


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)
    gold_questions = json.loads(GOLD_QUESTIONS_PATH.read_text(encoding="utf-8"))
    if args.limit is not None:
        gold_questions = gold_questions[: args.limit]

    gemini_callable = _SharedGeminiThrottle(
        make_gemini_callable(), args.min_call_interval
    )
    results = [_evaluate_question(gold, gemini_callable) for gold in gold_questions]
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
