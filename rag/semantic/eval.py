"""Reusable evaluation harness for the 50-question benchmark (eval phase).

Dataset format (JSON): {"version": ..., "cases": [EvalCase...]} where each
case carries id/difficulty/source/question, optional conversation_context,
optional stub payloads, and expected behavior constraints. The harness runs
each case with stubbed providers (no live quota), records stage-level
correctness (intent/resolution/query/fact/safe-failure), and assigns one
failure category per case. Results stream as JSONL for machine comparison.

Teacher-slide rows: the harness can import candidate rows from
ground_truth/rag/gold_questions.json by ID with source
"gold_candidate_pending". Which gold rows are verbatim teacher-slide
wording requires human confirmation; nothing is auto-labeled teacher_slide.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVAL_DIFFICULTIES = frozenset({"easy", "medium", "hard"})

EVAL_SOURCES = frozenset({"teacher_slide", "custom", "gold_candidate_pending"})

STAGE_CHECKS = (
    "intent_correct",
    "resolution_correct",
    "query_correct",
    "fact_correct",
    "safe_failure_correct",
)


def match_constraints(actual: Any, expected: Any) -> bool:
    """Recursive subset match: every expected leaf must equal the actual leaf."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and match_constraints(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        return all(match_constraints(a, e) for a, e in zip(actual, expected))
    return actual == expected


@dataclass(slots=True)
class EvalCaseResult:
    case_id: str
    difficulty: str | None
    source: str | None
    status: str | None
    checks: dict[str, bool]
    failure_category: str
    llm_request_count: int
    latency_ms: float


def _stub_provider(payload: Any) -> Any:
    def provide(_prompt: str) -> str:
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False)

    return provide


def run_eval_case(
    db_path: str | Path,
    case: dict[str, Any],
) -> EvalCaseResult:
    """Run one dataset case with stubbed providers and stage-graded checks."""
    from rag.semantic.pipeline import semantic_answer

    case_id = str(case.get("id", "unknown"))
    expected = case.get("expected", {}) if isinstance(case.get("expected"), dict) else {}
    stubs = case.get("stubs", {}) if isinstance(case.get("stubs"), dict) else {}
    outcome = semantic_answer(
        db_path,
        case.get("question", ""),
        case.get("conversation_context"),
        interpret_callable=_stub_provider(stubs.get("interpretation", "{}"))
        if "interpretation" in stubs
        else None,
        answer_callable=_stub_provider(stubs.get("answer", ""))
        if "answer" in stubs
        else None,
        sql_callable=_stub_provider(stubs.get("sql", ""))
        if "sql" in stubs
        else None,
    )
    trace = outcome.trace.to_dict()
    checks: dict[str, bool] = {}
    intent_expect = expected.get("intent")
    checks["intent_correct"] = (
        match_constraints(trace.get("semantic_intent"), intent_expect)
        if intent_expect is not None
        else True
    )
    resolution_expect = expected.get("resolved")
    checks["resolution_correct"] = (
        match_constraints(trace.get("resolved_intent"), resolution_expect)
        if resolution_expect is not None
        else True
    )
    query_expect = expected.get("plan")
    checks["query_correct"] = (
        match_constraints(trace.get("query_plan"), query_expect)
        if query_expect is not None
        else True
    )
    status_expect = expected.get("status")
    safe_expect = expected.get("safe_failure")
    if safe_expect is True:
        status_ok = outcome.result.status != "answer"
        if status_expect is not None:
            status_ok = status_ok and outcome.result.status == status_expect
    else:
        status_ok = True if status_expect is None else outcome.result.status == status_expect
    facts_expect = expected.get("facts_contain")
    facts_ok = True
    if facts_expect is not None:
        haystack = " ".join(
            [outcome.result.final_answer]
            + [str(fact) for fact in trace.get("verified_summary", {}).get("missing", [])]
        )
        facts_ok = all(str(item) in haystack for item in facts_expect)
    checks["fact_correct"] = bool(status_ok and facts_ok)
    if safe_expect is True:
        checks["safe_failure_correct"] = bool(
            outcome.result.status != "answer"
            and not outcome.result.claims
            and not outcome.result.provenance
        )
    else:
        checks["safe_failure_correct"] = True
    if all(checks.values()):
        failure_category = "NONE"
    elif not checks["intent_correct"]:
        failure_category = "INTERPRETATION_ERROR"
    elif not checks["resolution_correct"]:
        failure_category = "RESOLUTION_ERROR"
    elif not checks["query_correct"]:
        failure_category = "QUERY_ERROR"
    elif not checks["fact_correct"]:
        failure_category = (
            trace.get("failure_category")
            if trace.get("failure_category") not in (None, "NONE")
            else "GROUNDING_ERROR"
        )
    else:
        failure_category = "EXPECTED_SAFE_FAILURE"
    return EvalCaseResult(
        case_id=case_id,
        difficulty=case.get("difficulty"),
        source=case.get("source"),
        status=outcome.result.status,
        checks=checks,
        failure_category=failure_category,
        llm_request_count=trace.get("llm_request_count", 0),
        latency_ms=float(trace.get("timing", {}).get("total_ms", 0.0)),
    )


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases", []) if isinstance(data, dict) else []
    return [case for case in cases if isinstance(case, dict)]


def import_gold_candidates(
    gold_path: str | Path, ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """Import gold rows as pending-candidate cases (NOT teacher_slide)."""
    data = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("questions", [])
    cases: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if ids is not None and row.get("id") not in ids:
            continue
        cases.append(
            {
                "id": f"gold-{row.get('id', 'unknown')}",
                "difficulty": "medium",
                "source": "gold_candidate_pending",
                "question": row.get("question", ""),
                "expected": {},
            }
        )
    return cases


def run_evaluation(
    db_path: str | Path,
    dataset_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run every dataset case and stream machine-readable JSONL results."""
    cases = load_dataset(dataset_path)
    summary = {"total": 0, "passed": 0, "by_category": {}, "by_difficulty": {}}
    with open(output_path, "w", encoding="utf-8") as handle:
        for case in cases:
            result = run_eval_case(db_path, case)
            passed = all(result.checks.values())
            summary["total"] += 1
            if passed:
                summary["passed"] += 1
            summary["by_category"][result.failure_category] = (
                summary["by_category"].get(result.failure_category, 0) + 1
            )
            difficulty = result.difficulty or "unknown"
            difficulty_stats = summary["by_difficulty"].setdefault(
                difficulty, {"total": 0, "passed": 0}
            )
            difficulty_stats["total"] += 1
            if passed:
                difficulty_stats["passed"] += 1
            handle.write(
                json.dumps(
                    {
                        "id": result.case_id,
                        "difficulty": result.difficulty,
                        "source": result.source,
                        "status": result.status,
                        "checks": result.checks,
                        "failure_category": result.failure_category,
                        "llm_request_count": result.llm_request_count,
                        "latency_ms": result.latency_ms,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python -m rag.semantic.eval <db> <dataset.json> <out.jsonl>")
        raise SystemExit(2)
    print(
        json.dumps(
            run_evaluation(sys.argv[1], sys.argv[2], sys.argv[3]),
            ensure_ascii=False,
            indent=2,
        )
    )


__all__ = [
    "EVAL_DIFFICULTIES",
    "EVAL_SOURCES",
    "STAGE_CHECKS",
    "EvalCaseResult",
    "import_gold_candidates",
    "load_dataset",
    "match_constraints",
    "run_eval_case",
    "run_evaluation",
]
