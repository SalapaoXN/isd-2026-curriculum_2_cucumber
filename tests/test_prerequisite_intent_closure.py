import json
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.hard_qa import answer_hard_question
from rag.qa import ask
from rag.resolution import QueryContext
from tests import test_llm_sql_api as api_tests


DB = Path(__file__).parents[1] / "cucumber_outputs/runtime/curriculum.db"


class PrerequisiteIntentClosureTests(unittest.TestCase):
    def test_similar_repeated_collections_requery_without_stale_course(self):
        harness = api_tests.LlmSqlApiTests()
        harness.setUp()
        self.addCleanup(harness.tearDown)
        context = {"program": "AIT", "catalog_key": "ait-2566", "plan": "default",
                   "course_code": "06046401", "operations": ["placement"],
                   "semantic_topic": "cloud", "years": [1]}
        for question in (
            "วิชาใดบ้างที่มีวิชาบังคับก่อน บอกชื่อและวิชามา",
            "วิชาใดบ้างที่มีวิชาบังคับก่อน บอกชื่อและรหัสวิชามา",
            "วิชาใดบ้างที่มีวิชาบังคับก่อน",
            "วิชาใดบ้างที่มีวิชาบังคับก่อน",
        ):
            answer = harness._ask_with_stub_provider(question, context)
            self.assertEqual(answer["status"], "insufficient_evidence")
            self.assertEqual(answer["route"], "llm_sql")
            self.assertEqual(answer["provenance"], [])
            self.assertNotIn("06046405", answer["answer"])
            self.assertNotIn("06046406", answer["answer"])
            context = answer["next_context"]
            self.assertNotIn("semantic_topic", context)
            self.assertNotIn("course_code", context)
            self.assertNotIn("result_courses", context)

    def test_relationship_requests_do_not_call_hard_interpreter(self):
        for question in (
            "วิชาใดบ้างที่มีวิชาบังคับก่อน บอกชื่อและรหัสวิชามา",
            "วิชาไหนต้องผ่านวิชาอะไรมาก่อน",
            "06016414 มีวิชาบังคับก่อนไหม",
            "วิชาไหนมี prerequisite",
            "List courses with prerequisites ordered by course code",
        ):
            with self.subTest(question=question):
                provider = unittest.mock.Mock(side_effect=AssertionError("ordinary relationship request"))
                self.assertIsNone(answer_hard_question(DB, question, {"program": "AIT", "catalog_key": "ait-2566", "plan": "default"}, provider))
                provider.assert_not_called()

    def test_singular_prerequisite_followup_keeps_exact_course(self):
        harness = api_tests.LlmSqlApiTests()
        harness.setUp()
        self.addCleanup(harness.tearDown)
        context = {"program": "AIT", "catalog_key": "ait-2566", "plan": "default", "course_code": "06046401"}
        answer = harness._ask_with_stub_provider("วิชานี้มีวิชาบังคับก่อนอะไรบ้าง", context)
        self.assertEqual(answer["status"], "answer")
        self.assertIn("06046400", answer["answer"])
        self.assertNotIn("06046405", answer["answer"])
        self.assertTrue(answer["provenance"])

    def test_sequence_requests_reach_hard_interpreter(self):
        for question in ("ตรวจสอบลำดับวิชาบังคับก่อน", "แผนนี้จัดลำดับ prerequisite ถูกไหม"):
            with self.subTest(question=question):
                provider = unittest.mock.Mock(return_value="not JSON")
                result = answer_hard_question(DB, question, {"program": "AIT", "catalog_key": "ait-2566", "plan": "default"}, provider)
                self.assertIsNotNone(result)
                provider.assert_called_once()

    def test_canonical_relationship_collection_fails_closed_on_unknown_candidates(self):
        context = QueryContext(program="AIT", catalog_key="ait-2566", plan="default")
        response = ask(DB, "วิชาใดบ้างที่มีวิชาบังคับก่อน บอกชื่อและรหัสวิชามา", conversation_context=context)
        result = response["result"]
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertFalse(result.provenance)
        self.assertTrue(
            not result.claims
            or all(
                claim.operation == "list"
                and claim.status == "insufficient_evidence"
                and claim.value is None
                for claim in result.claims
            )
        )
        self.assertNotIn("06046401", result.final_answer)
        self.assertNotIn("06046400", result.final_answer)

    def test_plan_change_collection_fails_closed_independently_in_each_scope(self):
        harness = api_tests.LlmSqlApiTests()
        harness.setUp()
        self.addCleanup(harness.tearDown)
        question = "วิชาใดบ้างที่มีวิชาบังคับก่อน"
        first = harness._ask_with_stub_provider(question, {"program": "IT", "catalog_key": "it-2565", "plan": "coop"})
        self.assertEqual(first["status"], "insufficient_evidence")
        self.assertEqual(first["provenance"], [])
        self.assertNotIn("result_courses", first.get("next_context", {}))
        second = harness._ask_with_stub_provider(question, {"program": "IT", "catalog_key": "it-2565", "plan": "no_coop"})
        self.assertEqual(second["status"], "insufficient_evidence")
        self.assertEqual(second["provenance"], [])
        self.assertEqual(second["next_context"]["plan"], "no_coop")
        self.assertNotIn("แผนสหกิจ", second["answer"])
        self.assertNotIn("result_courses", second["next_context"])

    def test_sequence_presentation_preserves_status_and_separate_provenance(self):
        proposal = {"task_type": "prerequisite_sequence", "program": "AIT", "plan": "default", "left_plan": None,
                    "right_plan": None, "target_course_code": None, "horizon_terms": None}
        result = answer_hard_question(DB, "ตรวจสอบลำดับวิชาบังคับก่อนของหลักสูตรนี้",
                                     {"program": "AIT", "catalog_key": "ait-2566", "plan": "default"},
                                     lambda _: json.dumps(proposal))
        self.assertEqual(result["hard_task_type"], "prerequisite_sequence")
        self.assertTrue(result["provenance"])
        for text in ("default", "prerequisite", "อ้างอิง:"):
            self.assertNotIn(text, result["answer"])
        self.assertIn("หลักสูตร AIT", result["answer"])


if __name__ == "__main__":
    unittest.main()
