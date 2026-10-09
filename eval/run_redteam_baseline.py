"""Run the held-out red-team fixture and record the behavior baseline.

Deterministic: stubbed interpreter/answer/SQL providers, real runtime DB.
No prompt is modified; this script only observes. Output:
eval/results/final_question_redteam_baseline_v1.json
"""

import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.main import ask  # noqa: E402
from backend.schemas import AskRequest  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from rag.semantic.modes import QA_MODE_SEMANTIC  # noqa: E402
from rag.semantic.modes import semantic_ask_response  # noqa: E402

DB = ROOT / "cucumber_outputs" / "runtime" / "curriculum.db"
FIXTURE = ROOT / "eval" / "final_question_redteam_v1.json"
OUT = ROOT / "eval" / "results" / "final_question_redteam_baseline_v1.json"

SAFE_OUTCOMES = frozenset({
    "clarification_required", "insufficient_evidence", "unsupported",
    "scope_conflict", "context_conflict",
})


def _stub_callable(spec, *, kind):
    if spec is None:
        return None
    if spec == "TIMEOUT":
        def _raise(prompt, **kwargs):
            raise TimeoutError("redteam simulated provider timeout")
        return _raise
    if spec == "RAISE":
        def _boom(prompt, **kwargs):
            raise RuntimeError("redteam simulated provider failure")
        return _boom
    if isinstance(spec, dict):
        payload = json.dumps(spec, ensure_ascii=False)

        def _fixed(prompt, **kwargs):
            return payload
        return _fixed
    if isinstance(spec, str):
        def _text(prompt, **kwargs):
            return spec
        return _text
    raise ValueError(f"bad stub spec for {kind}")


def _run_pipeline(case):
    stubs = case.get("stubs", {})
    before = copy.deepcopy(case.get("ctx"))
    kwargs = {}
    if case.get("res") is not None:
        kwargs["clarification_resolution"] = case["res"]
    if case.get("ress") is not None:
        kwargs["clarification_resolutions"] = case["ress"]
    try:
        response = semantic_ask_response(
            DB, case["q"], copy.deepcopy(case.get("ctx")),
            home_program=case.get("home"),
            interpret_callable=_stub_callable(
                stubs.get("interpretation"), kind="interpret"),
            answer_callable=_stub_callable(stubs.get("answer"), kind="answer"),
            sql_callable=_stub_callable(
                stubs.get("interpretation"), kind="sql"),
            **kwargs,
        )
    except Exception as error:  # noqa: BLE001 - baseline must record, not crash
        try:
            from rag.semantic.errors import SemanticOperationalError
        except ImportError:
            SemanticOperationalError = ()
        if isinstance(error, SemanticOperationalError):
            return {"actual_status": error.status,
                    "actual_action": error.status,
                    "answer": "", "next_context": None,
                    "input_mutated": None}
        return {"actual_status": f"RAISED:{type(error).__name__}",
                "actual_action": None, "answer": "",
                "next_context": None, "input_mutated": None,
                "error": f"{type(error).__name__}"}
    mutated = before != case.get("ctx")
    return {"actual_status": response.get("status"),
            "actual_action": response.get("action"),
            "answer": response.get("answer") or "",
            "next_context": response.get("next_context"),
            "input_mutated": mutated}


def _run_api(case):
    stubs = case.get("stubs", {}) if "raw_request" not in case else {}
    providers = {
        "interpret_callable": _stub_callable(stubs.get("interpretation"),
                                            kind="interpret"),
        "answer_callable": _stub_callable(stubs.get("answer"), kind="answer"),
        "sql_callable": _stub_callable(stubs.get("interpretation"),
                                      kind="sql"),
    }
    with (patch("backend.main.active_qa_mode", return_value=QA_MODE_SEMANTIC),
          patch("backend.main._curriculum_db", return_value=DB),
          patch("backend.main._semantic_providers", return_value=providers)):
        try:
            if "raw_request" in case:
                request = AskRequest.model_validate(case["raw_request"])
            else:
                request = AskRequest(
                    question=case["q"],
                    conversation_context=copy.deepcopy(case.get("ctx")),
                    home_program=case.get("home"),
                )
        except ValidationError:
            return {"actual_status": "schema_reject", "actual_action": None,
                    "answer": "", "next_context": None,
                    "input_mutated": None}
        try:
            response = ask(request)
        except ValidationError:
            return {"actual_status": "schema_reject", "actual_action": None,
                    "answer": "", "next_context": None,
                    "input_mutated": None}
        except Exception as error:  # noqa: BLE001
            return {"actual_status": f"RAISED:{type(error).__name__}",
                    "actual_action": None, "answer": "",
                    "next_context": None, "input_mutated": None,
                    "error": f"{type(error).__name__}"}
    if not isinstance(response, dict):
        return {"actual_status": "NON_DICT_RESPONSE", "actual_action": None,
                "answer": "", "next_context": None,
                "input_mutated": None}
    return {"actual_status": response.get("status"),
            "actual_action": response.get("action"),
            "answer": response.get("answer") or "",
            "next_context": response.get("next_context"),
            "input_mutated": None}


def _suggest(case, record):
    expected = case["exp"]["outcome"]
    actual = record["actual_status"]
    violations = []
    answer = record["answer"] or ""
    if case["factual"]:
        for token in case["exp"].get("must_contain", ()):
            if token not in answer:
                violations.append(f"missing:{token}")
    for token in case["exp"].get("must_not_contain", ()):
        if token and token in answer:
            violations.append(f"forbidden-present:{token}")
    if expected == actual and not violations:
        return "SUGGEST_PASS", violations
    if expected == "safe_failure" and actual in SAFE_OUTCOMES and not violations:
        return "SUGGEST_SAFE", violations
    return "SUGGEST_REVIEW", violations


def main():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rows = []
    for case in fixture["cases"]:
        if case.get("layer") == "api":
            record = _run_api(case)
        else:
            record = _run_pipeline(case)
        suggestion, violations = _suggest(case, record)
        rows.append({"id": case["id"], "cat": case["cat"],
                     "expected": case["exp"]["outcome"],
                     "factual": case["factual"], **record,
                     "violations": violations, "suggestion": suggestion})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"version": fixture.get("version"),
                               "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    counts = {}
    for row in rows:
        counts[row["suggestion"]] = counts.get(row["suggestion"], 0) + 1
    print(f"cases: {len(rows)} {counts}")
    for row in rows:
        if row["suggestion"] == "SUGGEST_REVIEW":
            print(f"{row['suggestion']} {row['id']} exp={row['expected']} "
                  f"got={row['actual_status']} viol={row['violations']}")
            print(f"  ans: {(row['answer'] or '')[:160]}")
            if row.get("next_context") is not None:
                print(f"  ctx: {json.dumps(row['next_context'], ensure_ascii=False)[:200]}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
