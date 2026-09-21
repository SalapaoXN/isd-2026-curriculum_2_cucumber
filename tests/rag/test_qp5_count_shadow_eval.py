import json
import unittest
from pathlib import Path

from rag.qa import (
    _classify_structured_parse_completeness,
    _count_shadow_comparison,
    _is_count_shadow_candidate,
    _run_count_shadow,
)
from rag.query_spec import parse_query_spec
from rag.resolution import resolve_query_spec


DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


def _payload(
    *,
    program="IT",
    plans=(),
    years=(),
    semesters=(),
    **overrides,
):
    value = {
        "intent": "count_query",
        "proposed_program": program,
        "proposed_plans": list(plans),
        "proposed_years": list(years),
        "proposed_semesters": list(semesters),
        "course_codes": [],
        "topic": None,
        "requested_facts": ["course_list"],
        "judgement_dimension": None,
        "unresolved": [],
    }
    value.update(overrides)
    return json.dumps(value, ensure_ascii=False)


class CountShadowOfflineEvaluationTest(unittest.TestCase):
    def test_bounded_count_shadow_fixture_metrics(self):
        fixture = (
            ("IT ปี 3 ต้องเรียนกี่วิชา", False, None),
            ("IT ปี 3 มีรายวิชาทั้งหมดเท่าไหร่", True, None),
            ("IT ปี 3 มีรายวิชาทั้งหมดเท่าไร", True, None),
            (
                "IT แผนสหกิจ ปี 5 เทอม 1 วิชาเลือก มีรายวิชาทั้งหมดเท่าไหร่",
                True,
                {"plans": ("coop",), "years": (5,), "semesters": (1,)},
            ),
            ("IT มีวิชาเกี่ยวกับ data กี่วิชา", False, None),
            ("IT แผนสหกิจกับไม่สหกิจต่างกันยังไง", False, None),
            ("ปี 3 มีรายวิชาทั้งหมดเท่าไหร่", False, None),
            ("IT ปี 2 มีวิชาบังคับกี่วิชา", False, None),
            ("IT ปี 2 รวมกี่หน่วยกิต", False, None),
            ("IT 06016420 เรียนช่วงไหน", False, None),
        )

        predicted_eligible = 0
        true_positive = 0
        false_positive_calls = 0
        valid_attempts = 0
        valid_interpretations = 0
        query_spec_agreements = 0
        for question, expected, scope in fixture:
            spec = parse_query_spec(question)
            resolution = resolve_query_spec(spec, DB_PATH)
            completeness = _classify_structured_parse_completeness(
                spec,
                resolution,
            )
            actual = _is_count_shadow_candidate(spec, completeness, resolution)
            predicted_eligible += int(actual)
            true_positive += int(actual == expected and expected)
            calls = []
            if actual:
                scope = scope or {
                    "plans": tuple(spec.plans),
                    "years": tuple(spec.years),
                    "semesters": tuple(spec.semesters),
                }
                proposal = _payload(
                    plans=scope["plans"],
                    years=scope["years"],
                    semesters=scope["semesters"],
                )
                result = _run_count_shadow(
                    question,
                    spec,
                    completeness,
                    resolution,
                    None,
                    lambda prompt, proposal=proposal: calls.append(prompt)
                    or proposal,
                )
                valid_attempts += 1
                valid_interpretations += int(result.status == "validated")
                query_spec_agreements += int(
                    result.status == "validated"
                    and result.compiled_spec.operations == ("count",)
                    and _count_shadow_comparison(
                        spec,
                        result.compiled_spec,
                        None,
                    )
                    == "compatible_extension"
                )
            false_positive_calls += int(not expected and bool(calls))
            self.assertEqual(len(calls), int(actual))

        self.assertEqual(predicted_eligible, 3)
        self.assertEqual(true_positive, 3)
        self.assertEqual(false_positive_calls, 0)
        self.assertEqual(valid_interpretations, valid_attempts)
        self.assertEqual(query_spec_agreements, valid_attempts)

    def test_scope_conflict_and_malformed_proposals_are_rejected_offline(self):
        question = "IT ปี 3 มีรายวิชาทั้งหมดเท่าไหร่"
        spec = parse_query_spec(question)
        resolution = resolve_query_spec(spec, DB_PATH)
        completeness = _classify_structured_parse_completeness(spec, resolution)
        self.assertTrue(_is_count_shadow_candidate(spec, completeness, resolution))

        proposals = (
            _payload(plans=["coop"]),
            _payload(topic="data"),
            _payload(judgement_dimension="preference"),
            _payload(unresolved=["requirement_type"]),
            "not json",
        )
        attempted = 0
        rejected = 0
        max_calls = 0
        for proposal in proposals:
            calls = []
            result = _run_count_shadow(
                question,
                spec,
                completeness,
                resolution,
                None,
                lambda prompt, proposal=proposal: calls.append(prompt) or proposal,
            )
            attempted += int(result.attempted)
            rejected += int(result.status != "validated")
            max_calls = max(max_calls, len(calls))
        self.assertEqual(attempted, len(proposals))
        self.assertEqual(rejected, len(proposals))
        self.assertEqual(max_calls, 1)


if __name__ == "__main__":
    unittest.main()
