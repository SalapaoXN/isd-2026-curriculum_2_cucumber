"""QP3 offline evaluation of placement shadow interpretation.

Diagnostic-only gate before QP4/gated execution.  Uses mocks/fixtures only;
no real API is required for the core gate.

Production contract (must hold):
- shadow never affects real answers,
- shadow QuerySpec is never executed,
- planner/executor/retrieval/policy are not modified (only patched as
  spies inside these tests),
- no prerequisite/count/comparison family expansion.

Metrics measured separately:
1. eligibility precision
2. interpretation validity
3. compiled QuerySpec agreement with expected semantics
4. scope-conflict rejection
5. false-positive shadow calls on non-placement queries
6. max model-call count
7. answer equivalence with shadow OFF vs ON
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from rag import qa as qa_module
from rag.qa import (
    _classify_structured_parse_completeness,
    _placement_shadow_candidate,
    ask,
)
from rag.query_spec import parse_query_spec
from rag.resolution import resolve_query_spec


DB_PATH = str(
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


def placement_payload(**overrides):
    values = {
        "intent": "placement_query",
        "proposed_program": "DSBA",
        "proposed_plans": [],
        "proposed_years": [],
        "proposed_semesters": [],
        "course_codes": ["06026212"],
        "topic": None,
        "requested_facts": ["placement"],
        "judgement_dimension": None,
        "unresolved": [],
    }
    values.update(overrides)
    return json.dumps(values, ensure_ascii=False)


def valid_payload_for(question):
    """Build a scope-faithful placement proposal from deterministic parse."""
    spec = parse_query_spec(question)
    return placement_payload(
        proposed_program=spec.program,
        proposed_plans=list(spec.plans),
        proposed_years=list(spec.years),
        proposed_semesters=list(spec.semesters),
        course_codes=list(spec.course_codes),
    )


def counting_model(responder, counter):
    def _model(prompt):
        counter["calls"] += 1
        if len(counter.get("prompts", ())) < 4:
            counter.setdefault("prompts", []).append(prompt)
        return responder(prompt) if callable(responder) else responder

    return _model


# ---------------------------------------------------------------------------
# Fixture dataset.  Expected eligibility was verified against the real
# deterministic parser + resolver + bounded placement seam (offline probe).
# ---------------------------------------------------------------------------
ELIGIBILITY_CASES = (
    # deterministic-complete placement queries (must NOT shadow)
    {
        "id": "det-complete-canonical",
        "question": "DSBA \u0e27\u0e34\u0e0a\u0e32 06026212 \u0e40\u0e23\u0e35\u0e22\u0e19\u0e1b\u0e35\u0e44\u0e2b\u0e19 \u0e40\u0e17\u0e2d\u0e21\u0e44\u0e2b\u0e19",
        "eligible": False,
        "category": "deterministic-complete",
    },
    {
        "id": "det-complete-ตอนไหน",
        "question": "DSBA 06026212 \u0e40\u0e23\u0e35\u0e22\u0e19\u0e15\u0e2d\u0e19\u0e44\u0e2b\u0e19",
        "eligible": False,
        "category": "deterministic-complete",
    },
    {
        "id": "det-complete-year-scoped",
        "question": "IT \u0e1b\u0e35 3 \u0e27\u0e34\u0e0a\u0e32 06016414 \u0e40\u0e23\u0e35\u0e22\u0e19\u0e1b\u0e35\u0e44\u0e2b\u0e19 \u0e40\u0e17\u0e2d\u0e21\u0e44\u0e2b\u0e19",
        "eligible": False,
        "category": "deterministic-complete",
    },
    # placement long-tail queries eligible for shadow
    {
        "id": "longtail-สามารถลงได้ช่วงไหน",
        "question": "DSBA 06026212 \u0e2a\u0e32\u0e21\u0e32\u0e23\u0e16\u0e25\u0e07\u0e44\u0e14\u0e49\u0e0a\u0e48\u0e27\u0e07\u0e44\u0e2b\u0e19\u0e1a\u0e49\u0e32\u0e07",
        "eligible": True,
        "category": "long-tail",
    },
    {
        "id": "longtail-ลงเรียนช่วงไหน",
        "question": "DSBA 06026212 \u0e25\u0e07\u0e40\u0e23\u0e35\u0e22\u0e19\u0e0a\u0e48\u0e27\u0e07\u0e44\u0e2b\u0e19",
        "eligible": True,
        "category": "long-tail",
    },
    # paraphrases / wording variants
    {
        "id": "paraphrase-อยู่ช่วงไหน",
        "question": "DSBA 06026212 \u0e2d\u0e22\u0e39\u0e48\u0e0a\u0e48\u0e27\u0e07\u0e44\u0e2b\u0e19",
        "eligible": True,
        "category": "paraphrase",
    },
    {
        "id": "paraphrase-polite-suffix",
        "question": "DSBA 06026212 \u0e2a\u0e32\u0e21\u0e32\u0e23\u0e16\u0e25\u0e07\u0e44\u0e14\u0e49\u0e0a\u0e48\u0e27\u0e07\u0e44\u0e2b\u0e19\u0e1a\u0e49\u0e32\u0e07\u0e04\u0e23\u0e31\u0e1a",
        "eligible": True,
        "category": "paraphrase",
    },
    {
        "id": "paraphrase-deterministic-boundary",
        "question": "DSBA 06026212 \u0e2a\u0e32\u0e21\u0e32\u0e23\u0e16\u0e25\u0e07\u0e44\u0e14\u0e49\u0e40\u0e17\u0e2d\u0e21\u0e44\u0e2b\u0e19",
        "eligible": False,
        "category": "paraphrase",
    },
    # plan/year/semester scoped placement (long-tail wording -> eligible)
    {
        "id": "scoped-plan",
        "question": "IT \u0e41\u0e1c\u0e19\u0e2a\u0e2b\u0e01\u0e34\u0e08 \u0e27\u0e34\u0e0a\u0e32 06016414 \u0e25\u0e07\u0e40\u0e23\u0e35\u0e22\u0e19\u0e0a\u0e48\u0e27\u0e07\u0e44\u0e2b\u0e19",
        "eligible": True,
        "category": "plan-scoped",
        "expected": {"program": "IT", "plans": ("coop",), "course_codes": ("06016414",)},
    },
    {
        "id": "scoped-year",
        "question": "IT \u0e1b\u0e35 3 \u0e27\u0e34\u0e0a\u0e32 06016414 \u0e25\u0e07\u0e40\u0e23\u0e35\u0e22\u0e19\u0e0a\u0e48\u0e27\u0e07\u0e44\u0e2b\u0e19",
        "eligible": True,
        "category": "year-scoped",
        "expected": {"program": "IT", "years": (3,), "course_codes": ("06016414",)},
    },
    {
        "id": "scoped-semester",
        "question": "IT \u0e40\u0e17\u0e2d\u0e21 2 \u0e27\u0e34\u0e0a\u0e32 06016414 \u0e25\u0e07\u0e40\u0e23\u0e35\u0e22\u0e19\u0e0a\u0e48\u0e27\u0e07\u0e44\u0e2b\u0e19",
        "eligible": True,
        "category": "semester-scoped",
        "expected": {"program": "IT", "semesters": (2,), "course_codes": ("06016414",)},
    },
    {
        "id": "scoped-year-semester",
        "question": "IT \u0e1b\u0e35 3 \u0e40\u0e17\u0e2d\u0e21 1 \u0e27\u0e34\u0e0a\u0e32 06016414 \u0e25\u0e07\u0e40\u0e23\u0e35\u0e22\u0e19\u0e0a\u0e48\u0e27\u0e07\u0e44\u0e2b\u0e19",
        "eligible": True,
        "category": "scoped",
        "expected": {
            "program": "IT",
            "years": (3,),
            "semesters": (1,),
            "course_codes": ("06016414",),
        },
    },
    # year 5
    {
        "id": "year-5",
        "question": "IT \u0e1b\u0e35 5 \u0e27\u0e34\u0e0a\u0e32 06016414 \u0e25\u0e07\u0e40\u0e23\u0e35\u0e22\u0e19\u0e0a\u0e48\u0e27\u0e07\u0e44\u0e2b\u0e19",
        "eligible": True,
        "category": "year-5",
        "expected": {"program": "IT", "years": (5,), "course_codes": ("06016414",)},
    },
    # ambiguous scope (must NOT shadow)
    {
        "id": "ambiguous-no-program",
        "question": "06026212 \u0e2a\u0e32\u0e21\u0e32\u0e23\u0e16\u0e25\u0e07\u0e44\u0e14\u0e49\u0e0a\u0e48\u0e27\u0e07\u0e44\u0e2b\u0e19\u0e1a\u0e49\u0e32\u0e07",
        "eligible": False,
        "category": "ambiguous-scope",
    },
    {
        "id": "ambiguous-no-code",
        "question": "DSBA \u0e2a\u0e32\u0e21\u0e32\u0e23\u0e16\u0e25\u0e07\u0e44\u0e14\u0e49\u0e0a\u0e48\u0e27\u0e07\u0e44\u0e2b\u0e19\u0e1a\u0e49\u0e32\u0e07",
        "eligible": False,
        "category": "ambiguous-scope",
    },
    {
        "id": "ambiguous-unsupported-judgement",
        "question": "DSBA 06026212 \u0e40\u0e23\u0e35\u0e22\u0e19\u0e22\u0e32\u0e01\u0e44\u0e2b\u0e21",
        "eligible": False,
        "category": "ambiguous-scope",
    },
    # non-placement negatives (must never trigger placement shadow)
    {
        "id": "neg-list-topic",
        "question": "IT \u0e21\u0e35\u0e27\u0e34\u0e0a\u0e32\u0e40\u0e01\u0e35\u0e48\u0e22\u0e27\u0e01\u0e31\u0e1a database \u0e2d\u0e30\u0e44\u0e23\u0e1a\u0e49\u0e32\u0e07",
        "eligible": False,
        "category": "non-placement",
    },
    {
        "id": "neg-prerequisite",
        "question": "BIT 06023312 \u0e15\u0e49\u0e2d\u0e07\u0e1c\u0e48\u0e32\u0e19\u0e27\u0e34\u0e0a\u0e32\u0e1a\u0e31\u0e07\u0e04\u0e31\u0e1a\u0e01\u0e48\u0e2d\u0e19\u0e2d\u0e30\u0e44\u0e23\u0e1a\u0e49\u0e32\u0e07",
        "eligible": False,
        "category": "non-placement",
    },
    {
        "id": "neg-compare",
        "question": "IT 06016414 \u0e01\u0e31\u0e1a 06016420 \u0e15\u0e48\u0e32\u0e07\u0e01\u0e31\u0e19\u0e22\u0e31\u0e07\u0e44\u0e07",
        "eligible": False,
        "category": "non-placement",
    },
    {
        "id": "neg-count",
        "question": "IT \u0e1b\u0e35 3 \u0e21\u0e35\u0e01\u0e35\u0e48\u0e27\u0e34\u0e0a\u0e32",
        "eligible": False,
        "category": "non-placement",
    },
    {
        "id": "neg-describe",
        "question": "IT \u0e27\u0e34\u0e0a\u0e32 06016414 \u0e04\u0e37\u0e2d\u0e2d\u0e30\u0e44\u0e23",
        "eligible": False,
        "category": "non-placement",
    },
)

BASE_LONGTAIL = "DSBA 06026212 \u0e2a\u0e32\u0e21\u0e32\u0e23\u0e16\u0e25\u0e07\u0e44\u0e14\u0e49\u0e0a\u0e48\u0e27\u0e07\u0e44\u0e2b\u0e19\u0e1a\u0e49\u0e32\u0e07"

CONFLICT_CASES = (
    {"id": "conflict-program", "overrides": {"proposed_program": "IT"}},
    {"id": "conflict-plan", "overrides": {"proposed_plans": ["coop"]}},
    {"id": "conflict-year", "overrides": {"proposed_years": [2]}},
    {"id": "conflict-semester", "overrides": {"proposed_semesters": [1]}},
    {"id": "hallucinated-code", "overrides": {"course_codes": ["06029999"]}},
    {"id": "unresolved", "overrides": {"unresolved": ["which plan"]}},
)

MALFORMED_RESPONDERS = (
    ("malformed-json", lambda _prompt: "not json"),
    (
        "wrong-intent-family",
        lambda _prompt: placement_payload(
            intent="prerequisite_query",
            course_codes=["06026212"],
            requested_facts=["prerequisite"],
        ),
    ),
    (
        "forbidden-field",
        lambda _prompt: json.dumps(
            {
                "intent": "placement_query",
                "proposed_program": "DSBA",
                "course_codes": ["06026212"],
                "requested_facts": ["placement"],
                "sql": "SELECT 1",
            }
        ),
    ),
    (
        "provider-failure",
        lambda _prompt: (_ for _ in ()).throw(RuntimeError("offline")),
    ),
)


def _result_status(result):
    if isinstance(result, dict):
        return result.get("status")
    return getattr(result, "status", None)


def _shadow_of(result):
    """ask() omits intent_shadow on early non-answer returns; treat as idle."""
    shadow = result.get("intent_shadow")
    if shadow is None:
        return None
    return shadow


def _assert_shadow_idle(testcase, result, counter, case_id):
    testcase.assertEqual(counter["calls"], 0, case_id)
    shadow = _shadow_of(result)
    if shadow is not None:
        testcase.assertFalse(shadow.attempted, case_id)
        testcase.assertFalse(shadow.eligible, case_id)


class Qp3PlacementShadowEvalTest(unittest.TestCase):
    maxDiff = None

    # 1. eligibility precision -------------------------------------------
    def test_eligibility_precision(self):
        correct = 0
        errors = []
        for case in ELIGIBILITY_CASES:
            with self.subTest(case=case["id"]):
                spec = parse_query_spec(case["question"])
                resolution = resolve_query_spec(spec, DB_PATH)
                completeness = _classify_structured_parse_completeness(
                    spec, resolution, None
                )
                eligible = _placement_shadow_candidate(spec, completeness, resolution)
                if bool(eligible) == bool(case["eligible"]):
                    correct += 1
                else:
                    errors.append(case["id"])
                self.assertEqual(
                    bool(eligible),
                    bool(case["eligible"]),
                    f"eligibility mismatch for {case['id']}",
                )
        precision = correct / len(ELIGIBILITY_CASES)
        self.assertEqual(precision, 1.0, f"eligibility errors: {errors}")

    # 2. interpretation validity ------------------------------------------
    def test_interpretation_validity(self):
        # valid proposals validate; every malformed/conflicting proposal fails closed
        counter = {"calls": 0}
        model = counting_model(lambda _p: valid_payload_for(BASE_LONGTAIL), counter)
        result = ask(
            DB_PATH,
            BASE_LONGTAIL,
            intent_model_callable=model,
            shadow_intent=True,
        )
        shadow = result["intent_shadow"]
        self.assertTrue(shadow.attempted)
        self.assertTrue(shadow.eligible)
        self.assertEqual(shadow.status, "validated")
        self.assertEqual(shadow.comparison, "compatible_extension")

        for case in CONFLICT_CASES:
            with self.subTest(case=case["id"]):
                result = ask(
                    DB_PATH,
                    BASE_LONGTAIL,
                    intent_model_callable=lambda _p, o=case["overrides"]: placement_payload(**o),
                    shadow_intent=True,
                )
                self.assertEqual(
                    result["intent_shadow"].status,
                    "invalid_interpretation",
                    case["id"],
                )

        for name, responder in MALFORMED_RESPONDERS:
            with self.subTest(case=name):
                result = ask(
                    DB_PATH,
                    BASE_LONGTAIL,
                    intent_model_callable=counting_model(responder, {"calls": 0}),
                    shadow_intent=True,
                )
                self.assertIn(
                    result["intent_shadow"].status,
                    {"invalid_interpretation", "unavailable"},
                    name,
                )
                self.assertNotEqual(result["intent_shadow"].status, "validated", name)

    # 3. compiled QuerySpec agreement -------------------------------------
    def test_compiled_spec_agreement(self):
        scoped = [c for c in ELIGIBILITY_CASES if c.get("expected")]
        self.assertGreaterEqual(len(scoped), 4)
        for case in scoped:
            with self.subTest(case=case["id"]):
                counter = {"calls": 0}
                result = ask(
                    DB_PATH,
                    case["question"],
                    intent_model_callable=counting_model(
                        lambda _p, q=case["question"]: valid_payload_for(q), counter
                    ),
                    shadow_intent=True,
                )
                shadow = result["intent_shadow"]
                self.assertEqual(shadow.status, "validated", case["id"])
                compiled = shadow.compiled_spec
                self.assertIsNotNone(compiled, case["id"])
                self.assertEqual(compiled.operations, ("placement",), case["id"])
                self.assertEqual(compiled.judgement, "none", case["id"])
                self.assertIsNone(compiled.topic, case["id"])
                for field, expected in case["expected"].items():
                    actual = getattr(compiled, field)
                    if isinstance(expected, tuple):
                        self.assertEqual(tuple(actual), tuple(expected))
                    else:
                        self.assertEqual(actual, expected, f"{case['id']}.{field}")
                # authoritative axes preserved from deterministic parse
                spec = parse_query_spec(case["question"])
                self.assertEqual(compiled.program, spec.program, case["id"])
                self.assertEqual(tuple(compiled.course_codes), tuple(spec.course_codes))

        # year-5 boundary: deterministic year survives compilation
        year5 = next(c for c in ELIGIBILITY_CASES if c["id"] == "year-5")
        spec = parse_query_spec(year5["question"])
        self.assertEqual(spec.years, (5,))

    # 4. scope-conflict rejection ------------------------------------------
    def test_scope_conflict_rejection(self):
        rejected = 0
        for case in CONFLICT_CASES:
            with self.subTest(case=case["id"]):
                with patch(
                    "rag.qa.plan_evidence",
                    side_effect=AssertionError("conflict must not plan"),
                ), patch(
                    "rag.qa.execute_evidence_plan",
                    side_effect=AssertionError("conflict must not execute"),
                ):
                    result = ask(
                        DB_PATH,
                        BASE_LONGTAIL,
                        intent_model_callable=lambda _p, o=case[
                            "overrides"
                        ]: placement_payload(**o),
                        shadow_intent=True,
                    )
                shadow = result["intent_shadow"]
                self.assertEqual(shadow.status, "invalid_interpretation", case["id"])
                self.assertIn(
                    shadow.comparison, {"conflict", "invalid_interpretation"}, case["id"]
                )
                self.assertEqual(
                    _result_status(result["result"]), "insufficient_evidence", case["id"]
                )
                rejected += 1
        self.assertEqual(rejected, len(CONFLICT_CASES))

    # 5. false-positive shadow calls on non-placement queries -------------
    def test_false_positive_shadow_calls_on_nonplacement(self):
        negatives = [c for c in ELIGIBILITY_CASES if c["category"] == "non-placement"]
        self.assertGreaterEqual(len(negatives), 5)
        false_positives = 0
        for case in negatives:
            with self.subTest(case=case["id"]):
                counter = {"calls": 0}

                def _must_not_call(_prompt):
                    raise AssertionError(
                        f"non-placement query must not shadow: {case['id']}"
                    )

                orig_plan = qa_module.plan_evidence
                orig_exec = qa_module.execute_evidence_plan
                plan_calls = []
                exec_calls = []

                def _plan(spec, resolution, *args, **kwargs):
                    plan_calls.append(tuple(spec.operations))
                    return orig_plan(spec, resolution, *args, **kwargs)

                def _exec(db_path, plan, *args, **kwargs):
                    exec_calls.append(1)
                    return orig_exec(db_path, plan, *args, **kwargs)

                with patch("rag.qa.plan_evidence", side_effect=_plan), patch(
                    "rag.qa.execute_evidence_plan", side_effect=_exec
                ):
                    result = ask(
                        DB_PATH,
                        case["question"],
                        intent_model_callable=counting_model(_must_not_call, counter),
                        shadow_intent=True,
                    )
                _assert_shadow_idle(self, result, counter, case["id"])
                if counter["calls"]:
                    false_positives += 1
        self.assertEqual(false_positives, 0)

    # 6. max model-call count ----------------------------------------------
    def test_max_model_call_count(self):
        worst = 0
        for case in ELIGIBILITY_CASES:
            with self.subTest(case=case["id"]):
                counter = {"calls": 0}

                def _fail(_prompt):
                    raise AssertionError("must not be called")

                responder = (
                    (lambda _p, q=case["question"]: valid_payload_for(q))
                    if case["eligible"]
                    else _fail
                )
                result = ask(
                    DB_PATH,
                    case["question"],
                    intent_model_callable=counting_model(responder, counter),
                    shadow_intent=True,
                )
                self.assertLessEqual(counter["calls"], 1, case["id"])
                worst = max(worst, counter["calls"])
                if case["eligible"]:
                    self.assertEqual(counter["calls"], 1, case["id"])
                    self.assertTrue(result["intent_shadow"].attempted, case["id"])
                else:
                    _assert_shadow_idle(self, result, counter, case["id"])
        self.assertLessEqual(worst, 1)
        self.assertEqual(worst, 1)

    # 7. answer equivalence remains frozen for non-eligible shadow cases.
    # QP4 intentionally allows a validated compatible placement proposal to
    # execute, so eligible cases are checked for that explicit transition.
    def test_answer_equivalence_shadow_off_vs_on(self):
        mismatches = []
        for case in ELIGIBILITY_CASES:
            with self.subTest(case=case["id"]):
                orig_plan = qa_module.plan_evidence
                orig_exec = qa_module.execute_evidence_plan
                off_plan = []
                off_exec = []
                on_plan = []
                on_exec = []

                def _off_plan(spec, resolution, *args, **kwargs):
                    off_plan.append(tuple(spec.operations))
                    return orig_plan(spec, resolution, *args, **kwargs)

                def _off_exec(db_path, plan, *args, **kwargs):
                    off_exec.append(1)
                    return orig_exec(db_path, plan, *args, **kwargs)

                with patch("rag.qa.plan_evidence", side_effect=_off_plan), patch(
                    "rag.qa.execute_evidence_plan", side_effect=_off_exec
                ):
                    off = ask(DB_PATH, case["question"])

                def _on_plan(spec, resolution, *args, **kwargs):
                    on_plan.append(tuple(spec.operations))
                    return orig_plan(spec, resolution, *args, **kwargs)

                def _on_exec(db_path, plan, *args, **kwargs):
                    on_exec.append(1)
                    return orig_exec(db_path, plan, *args, **kwargs)

                counter = {"calls": 0}

                def _fail(_prompt):
                    raise AssertionError("non-eligible must use zero calls")

                responder = (
                    (lambda _p, q=case["question"]: valid_payload_for(q))
                    if case["eligible"]
                    else _fail
                )
                with patch("rag.qa.plan_evidence", side_effect=_on_plan), patch(
                    "rag.qa.execute_evidence_plan", side_effect=_on_exec
                ):
                    on = ask(
                        DB_PATH,
                        case["question"],
                        intent_model_callable=counting_model(responder, counter),
                        shadow_intent=True,
                    )
                self.assertEqual(on["route"], off["route"], case["id"])
                if case["eligible"]:
                    self.assertEqual(on["intent_shadow"].status, "validated", case["id"])
                    self.assertTrue(on["intent_shadow"].executed, case["id"])
                    self.assertIn(
                        _result_status(on["result"]),
                        ("answer", "valid_empty"),
                        case["id"],
                    )
                else:
                    self.assertEqual(on["result"], off["result"], case["id"])
                    if on["result"] != off["result"]:
                        mismatches.append(case["id"])
        self.assertEqual(mismatches, [])

    # QP4: validated compatible placement shadow executes through the
    # canonical deterministic placement seam exactly once.
    def test_validated_shadow_execution_impact(self):
        eligible = [c for c in ELIGIBILITY_CASES if c["eligible"]]
        self.assertGreaterEqual(len(eligible), 8)
        for case in eligible:
            with self.subTest(case=case["id"]):
                with patch("rag.qa.plan_evidence") as planner, patch(
                    "rag.qa.execute_evidence_plan"
                ) as executor:
                    result = ask(
                        DB_PATH,
                        case["question"],
                        intent_model_callable=counting_model(
                            lambda _p, q=case["question"]: valid_payload_for(q),
                            {"calls": 0},
                        ),
                        shadow_intent=True,
                    )
                self.assertEqual(
                    _result_status(result["result"]), "insufficient_evidence", case["id"]
                )
                planner.assert_called_once()
                executor.assert_called_once()
                self.assertEqual(result["intent_shadow"].status, "validated", case["id"])
                self.assertTrue(result["intent_shadow"].executed, case["id"])

    # optional real-provider seam: kept separate, never required for the gate --
    def test_optional_real_provider_seam(self):
        if os.getenv("QP3_REAL_MODEL") != "1":
            self.skipTest("offline gate only; set QP3_REAL_MODEL=1 to report separately")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.skipTest("QP3_REAL_MODEL=1 but GEMINI_API_KEY is absent")
        from rag.providers.gemini import make_gemini_callable

        model_callable = make_gemini_callable()
        result = ask(
            DB_PATH,
            BASE_LONGTAIL,
            intent_model_callable=model_callable,
            shadow_intent=True,
        )
        # Report-only: real-provider outcome must still be harmless.
        self.assertEqual(
            _result_status(result["result"]), "insufficient_evidence"
        )


if __name__ == "__main__":
    unittest.main()
