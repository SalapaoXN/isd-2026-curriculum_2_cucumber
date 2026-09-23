"""ROBUST-2: public-path evaluation of the real-user robustness set.

Supplemental confidence only — this runner must NOT redefine the Core
35 completion contract. Every case runs through rag.qa.ask() with
raising stubs (all robustness budgets are 0 calls). Failures are
classified A/B/C/D/E per the task policy; production code is never
edited from here.
"""

import json
import unittest
from pathlib import Path

from rag.qa import ask

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "real_user_robustness_v1.json"
DB_PATH = Path(__file__).resolve().parents[2] / "cucumber_outputs" / "runtime" / "curriculum.db"


def _raising_stub(calls, name):
    def _stub(prompt: str) -> str:
        calls.append(prompt)
        raise AssertionError(f"{name} model must not be called")
    return _stub


def _intent_payload(spec):
    return json.dumps(
        {
            "intent": spec["intent"],
            "proposed_program": spec.get("proposed_program"),
            "proposed_plans": spec.get("proposed_plans", []),
            "proposed_years": spec.get("proposed_years", []),
            "proposed_semesters": spec.get("proposed_semesters", []),
            "course_codes": spec.get("course_codes", []),
            "topic": spec.get("topic"),
            "requested_facts": spec["requested_facts"],
            "judgement_dimension": spec.get("judgement_dimension"),
            "unresolved": [],
        },
        ensure_ascii=False,
    )


def _claim_text(claim) -> str:
    return json.dumps(
        {"value": claim.value, "evidence": str(claim.evidence)[:500]},
        ensure_ascii=False,
        default=str,
    )


class RealUserRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.cases = doc["cases"]

    def _run_case(self, case):
        calls: dict[str, list[str]] = {"intent": [], "structured": [], "answer": []}
        stub = case.get("harness_stub") or {}
        if stub.get("intent"):
            intent_calls = calls["intent"]
            payload = _intent_payload(stub)

            def intent_stub(prompt: str, _p=payload, _c=intent_calls) -> str:
                _c.append(prompt)
                return _p
        else:
            self.assertEqual(case["budget"]["intent"], 0, f"{case['id']}: intent budget")
            intent_stub = _raising_stub(calls["intent"], "intent")
        context = None
        response = None
        for turn in case["turns"]:
            kwargs = {
                "structured_model_callable": _raising_stub(calls["structured"], "structured"),
                "answer_model_callable": None,
                "intent_model_callable": intent_stub,
            }
            if turn.get("context") == "previous":
                self.assertIsNotNone(context, f"{case['id']}: no context to reuse")
                kwargs["conversation_context"] = context
            response = ask(DB_PATH, turn["question"], **kwargs)
            context = response.get("next_context")
        result = response["result"]
        if isinstance(result, dict):
            status = result.get("status") or result.get("action")
        else:
            status = result.status
        self.assertEqual(status, case["expected_status"], f"{case['id']}: status")
        if isinstance(result, dict):
            self.assertEqual(case["claim_checks"], [], f"{case['id']}: blocked must be claimless")
            return
        for check in case["claim_checks"]:
            matches = [
                claim
                for claim in result.claims
                if claim.operation == check["operation"] and claim.status == check["status"]
            ]
            self.assertTrue(matches, f"{case['id']}: no claim {check}")
            if "value" in check:
                self.assertTrue(
                    any(claim.value == check["value"] for claim in matches),
                    f"{case['id']}: no claim value {check['value']}",
                )
            for text in check.get("contains", ()):
                self.assertTrue(
                    any(text in _claim_text(claim) or text in result.final_answer for claim in matches),
                    f"{case['id']}: {text!r} not in claim/final",
                )
            if "scope_credit" in check:
                self.assertTrue(
                    any(
                        getattr(claim.effective_scope, "credit_units", None) == check["scope_credit"]
                        for claim in matches
                    ),
                    f"{case['id']}: scope credit mismatch",
                )
        for text in case["final_contains"]:
            self.assertIn(text, result.final_answer, f"{case['id']}: final missing {text!r}")
        if case["provenance_required"]:
            self.assertTrue(result.provenance, f"{case['id']}: provenance missing")
        self.assertEqual(calls["intent"], [], f"{case['id']}: intent budget") if not stub.get("intent") else self.assertEqual(
            len(calls["intent"]), case["budget"]["intent"], f"{case['id']}: intent budget")
        if case.get("structured_attempt_ok"):
            self.assertLessEqual(len(calls["structured"]), case["budget"]["structured"],
                                 f"{case['id']}: structured budget")
        else:
            self.assertEqual(calls["structured"], [], f"{case['id']}: structured budget")

    def test_robustness_cases(self):
        for case in self.cases:
            with self.subTest(case_id=case["id"]):
                self._run_case(case)
        self.assertEqual(len(self.cases), 50)


if __name__ == "__main__":
    unittest.main()
