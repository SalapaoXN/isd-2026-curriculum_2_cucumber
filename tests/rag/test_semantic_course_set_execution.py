"""G4 canonical evidence, atomicity, presentation and context controls."""

import copy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from rag.semantic import executor
from rag.semantic.answerer import render_semantic_answer
from rag.semantic.pipeline import semantic_answer
from rag.semantic.schema import VerifiedResult
from rag.structured.queries import scoped_course_set
from tests.rag.test_semantic_course_set_contract import payload
from tests.rag.test_semantic_course_set_resolution import DB, resolve


def group_payload():
    return scoped_course_set(DB, "IT", ("coop",), catalog_key="it-2565",
                             course_targets=({"course_code": "06016481"}, {"course_code": "06016482"}))


class CourseSetExecutionTests(unittest.TestCase):
    def execute(self, resolved=None):
        return executor.execute_explicit_course_set(DB, resolved or resolve(fields=("placement", "alternative_selection")), "question")

    def test_canonical_exactly_one_of_two(self):
        result = self.execute()
        self.assertEqual(result.status, "answer", result.missing_information)
        self.assertEqual(len(result.alternative_selections), 1)
        fact = result.alternative_selections[0]
        self.assertEqual((fact.program, fact.catalog_key, fact.plan), ("IT", "it-2565", "coop"))
        self.assertEqual(fact.member_course_codes, ("06016481", "06016482"))
        self.assertEqual((fact.minimum_choices, fact.maximum_choices), (1, 1))
        self.assertTrue(fact.provenance)
        self.assertEqual([c["course_code"] for c in result.result_courses], ["06016481", "06016482"])
        self.assertTrue(all("ชั้นปีที่ 3" in line and "ภาคการศึกษาที่ 2" in line
                            for line in result.summary_facts if "ภาคการศึกษาที่" in line))

    def test_bad_group_evidence_fails_closed(self):
        base = group_payload()
        variants = []
        for key, value in (("alternative_group_id", None), ("minimum_choices", None),
                           ("maximum_choices", None), ("minimum_choices", True),
                           ("minimum_choices", -1), ("maximum_choices", 3),
                           ("minimum_choices", 2), ("provenance", []),
                           ("catalog_id", -1), ("plan_key", "no_coop")):
            data = copy.deepcopy(base)
            data["courses"][0][key] = value
            variants.append((key, data))
        data = copy.deepcopy(base)
        data["courses"][0]["alternative_courses"][0]["provenance"] = []
        variants.append(("member provenance", data))
        data = copy.deepcopy(base)
        data["courses"][0]["alternative_courses"][0]["catalog_id"] = -1
        variants.append(("member edition", data))
        for label, data in variants:
            with self.subTest(label=label), patch("rag.semantic.executor.scoped_course_set", return_value=data):
                result = self.execute()
                self.assertNotEqual(result.status, "answer")
                self.assertEqual(result.claims, ())
                self.assertEqual(result.alternative_selections, ())

    def test_subset_of_larger_group_is_not_exactly_one_of_two(self):
        data = group_payload()
        extra = copy.deepcopy(data["courses"][0]["alternative_courses"][0])
        extra["course_code"] = "06016413"
        data["courses"][0]["alternative_courses"].append(extra)
        with patch("rag.semantic.executor.scoped_course_set", return_value=data):
            self.assertNotEqual(self.execute().status, "answer")

    def test_conflicting_group_rows_rejected(self):
        for key, value in (("maximum_choices", 2), ("alternative_group_id", 999)):
            data = group_payload()
            row = copy.deepcopy(data["courses"][0])
            row[key] = value
            data["courses"].append(row)
            with self.subTest(key=key), patch("rag.semantic.executor.scoped_course_set", return_value=data):
                self.assertNotEqual(self.execute().status, "answer")

    def test_agreeing_duplicate_rows_deduplicate(self):
        data = group_payload()
        data["courses"].append(copy.deepcopy(data["courses"][0]))
        with patch("rag.semantic.executor.scoped_course_set", return_value=data):
            self.assertEqual(len(self.execute().alternative_selections), 1)

    def test_missing_group_relation_rejected(self):
        with patch("rag.semantic.executor.scoped_course_set", return_value={"status": "no_data", "courses": []}):
            self.assertNotEqual(self.execute().status, "answer")

    def test_merged_provenance_cannot_hide_missing_group_source(self):
        original = executor._provenance_for
        def sources(connection, table, column, identity):
            return [] if table == "alternative_group_provenance" else original(connection, table, column, identity)
        with patch("rag.semantic.executor._provenance_for", side_effect=sources):
            self.assertNotEqual(self.execute().status, "answer")

    def test_selection_only_uses_canonical_bounds(self):
        resolved = resolve(fields=("alternative_selection",))
        resolved = replace(resolved, intent=replace(resolved.intent, relation="alternative_selection"))
        self.assertEqual(self.execute(resolved).alternative_selections[0].minimum_choices, 1)

    def test_presentation_bound_fails_instead_of_truncating(self):
        from rag.semantic.answerer import MAX_ANSWER_LEN
        with patch("rag.semantic.answerer.render_verified_course_set", return_value="x" * (MAX_ANSWER_LEN + 1)):
            result = self.execute()
        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.claims, ())

    def test_member_failure_discards_preceding_success(self):
        original = executor.execute_deterministic
        calls = []
        def execute(db, spec, context, question):
            calls.append(spec.course_codes)
            if len(calls) == 2:
                return VerifiedResult(status="missing_data", missing_information=("missing member evidence",))
            return original(db, spec, context, question)
        with patch("rag.semantic.executor.execute_deterministic", side_effect=execute):
            result = self.execute(resolve(("06016413", "06016420", "06016421"), plan="no_coop"))
        self.assertNotEqual(result.status, "answer")
        self.assertEqual(result.claims, ())
        self.assertEqual(result.result_courses, ())
        self.assertEqual(len(calls), 2)

    def test_missing_requested_operation_blocks_whole_result(self):
        original = executor.execute_deterministic
        def execute(*args):
            result = original(*args)
            return replace(result, claims=tuple(c for c in result.claims if c.operation != "placement"))
        with patch("rag.semantic.executor.execute_deterministic", side_effect=execute):
            self.assertNotEqual(self.execute().status, "answer")

    def test_credits_execute_per_member_not_as_set_sum(self):
        result = self.execute(resolve(("06016413", "06016420"), ("credits",), plan="no_coop"))
        self.assertEqual(result.status, "answer", result.missing_information)
        self.assertEqual(sum(c.operation == "sum_credits" for c in result.claims), 2)
        self.assertEqual([c["course_code"] for c in result.result_courses], ["06016413", "06016420"])

    def test_prerequisites_bound_to_subject_member(self):
        result = self.execute(resolve(("06016413", "06016420", "06016421"),
                                      ("placement", "prerequisites"), plan="no_coop"))
        self.assertEqual(result.status, "answer", result.missing_information)
        prereqs = [line for line in result.summary_facts if "วิชาบังคับก่อน" in line]
        self.assertEqual(len(prereqs), 3)
        self.assertTrue(any(line.startswith("06016413") and "ไม่มีวิชาบังคับก่อน" in line for line in prereqs))
        for code in ("06016420", "06016421"):
            self.assertTrue(any(line.startswith(code) and ": 06016413" in line for line in prereqs))

    def test_selection_presentation_is_deterministic(self):
        result = self.execute()
        def forbidden(_prompt):
            self.fail("G4 set presentation must not call a provider")
        answer, mode = render_semantic_answer("question", result, forbidden)
        self.assertEqual(mode, "deterministic")
        self.assertIn("06016481", answer)
        self.assertIn("06016482", answer)
        self.assertIn("เลือก 1 วิชาจาก 2 วิชา", answer)
        self.assertIn("สหกิจ", answer)

    def test_pipeline_retains_set_without_first_member_focus(self):
        data = payload()
        data["scope"]["plan"] = "coop"
        question = "coop availability and selection for 06016481 and 06016482"
        result = semantic_answer(DB, question, {"program": "IT", "catalog_key": "it-2565"},
                                 interpret_callable=lambda _prompt: json.dumps(data))
        self.assertEqual(result.result.status, "answer", result.trace.failure_reason)
        self.assertNotIn("focus_course", result.next_context)
        self.assertEqual([c["course_code"] for c in result.next_context["result_courses"]], ["06016481", "06016482"])
        self.assertEqual(len(result.trace.semantic_intent["target"]["members"]), 2)

    def test_sequence_request_never_executes_weaker_lookup(self):
        data = payload(("06016413", "06016420", "06016421"),
                       ("placement", "prerequisites", "placement_sequence"))
        with patch("rag.semantic.pipeline.execute_deterministic") as evidence:
            result = semantic_answer(DB, "Arrange 06016413 06016420 06016421",
                                     {"program": "IT", "catalog_key": "it-2565", "plan": "no_coop"},
                                     interpret_callable=lambda _prompt: json.dumps(data))
        self.assertEqual(result.result.status, "unsupported")
        self.assertIn("placement_sequence", result.trace.failure_reason)
        self.assertEqual(len(result.trace.semantic_intent["target"]["members"]), 3)
        evidence.assert_not_called()

    def test_unresolved_member_never_executes_remaining_members(self):
        data = payload(("06016413", "06016420", "06019999"), ("placement",))
        with patch("rag.semantic.pipeline.execute_deterministic") as evidence:
            result = semantic_answer(DB, "Place 06016413 06016420 06019999",
                                     {"program": "IT", "catalog_key": "it-2565"},
                                     interpret_callable=lambda _prompt: json.dumps(data))
        self.assertNotEqual(result.result.status, "answer")
        self.assertEqual(result.result.claims, ())
        evidence.assert_not_called()


if __name__ == "__main__":
    unittest.main()
