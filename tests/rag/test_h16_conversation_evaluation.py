import dataclasses
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.grounded_answer import GroundedAnswerResult
from rag.qa import ask
from rag.resolution import QueryContext


DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


class ConversationContextEvaluationTests(unittest.TestCase):
    def _answer(self, question, *, context=None):
        result = ask(DB_PATH, question, conversation_context=context)
        self.assertIsInstance(result["result"], GroundedAnswerResult)
        self.assertEqual(result["result"].status, "answer")
        return result

    def test_semester_followup_requeries_and_replaces_scope(self):
        with patch("rag.qa.execute_evidence_plan", wraps=__import__(
            "rag.qa", fromlist=["execute_evidence_plan"]
        ).execute_evidence_plan) as execute:
            first = self._answer("IT ปี 3 เทอม 1 มีทั้งหมดกี่วิชา")
            second = self._answer(
                "แล้วเทอม 2 ล่ะ", context=first["next_context"]
            )

        self.assertEqual(execute.call_count, 2)
        self.assertTrue(all(
            claim.effective_scope.years == (3,)
            and claim.effective_scope.semesters == (2,)
            for claim in second["result"].claims
        ))
        self.assertNotEqual(first["result"].provenance, second["result"].provenance)

    def test_year_followup_replaces_year_and_preserves_semester(self):
        first = self._answer("IT ปี 3 เทอม 1 มีทั้งหมดกี่วิชา")
        second = self._answer("แล้วปี 2 ล่ะ", context=first["next_context"])
        self.assertTrue(all(
            claim.effective_scope.years == (2,)
            and claim.effective_scope.semesters == (1,)
            for claim in second["result"].claims
        ))

    def test_program_switch_does_not_leak_previous_program(self):
        first = self._answer("IT มีกี่วิชา")
        second = self._answer("แล้ว DSBA ล่ะ", context=first["next_context"])
        self.assertTrue(all(
            claim.effective_scope.program == "DSBA"
            for claim in second["result"].claims
        ))

    def test_exact_course_followups_requery_credit_prerequisite_and_placement(self):
        first = self._answer("IT 06016454 คือวิชาอะไร")
        context = first["next_context"]
        for question, operation in (
            ("แล้วกี่หน่วยกิต", "sum_credits"),
            ("มี prerequisite ไหม", "prerequisite"),
            ("วิชานี้เรียนตอนไหน", "placement"),
        ):
            with self.subTest(question=question):
                result = self._answer(question, context=context)
                self.assertIn(
                    operation,
                    [claim.operation for claim in result["result"].claims],
                )
                self.assertTrue(result["result"].provenance)

    def test_explicit_new_course_replaces_previous_course(self):
        first = self._answer("IT 06016454 คือวิชาอะไร")
        second = self._answer(
            "IT 06016455 มีกี่หน่วยกิต", context=first["next_context"]
        )
        self.assertTrue(all(
            "06016455" in claim.value
            for claim in second["result"].claims
            if isinstance(claim.value, str)
        ))
        self.assertFalse(any(
            "06016454" in claim.value
            for claim in second["result"].claims
            if isinstance(claim.value, str)
        ))

    def test_bare_then_does_not_invent_an_operation(self):
        context = QueryContext(program="IT", years=(3,), operations=("count",))
        result = ask(DB_PATH, "แล้ว", conversation_context=context)
        self.assertNotEqual(result.get("route"), "structured")
        self.assertNotEqual(
            getattr(result.get("result"), "status", None), "answer"
        )

    def test_topic_turn_does_not_create_structured_context(self):
        result = self._answer("IT มีวิชาเกี่ยวกับ database อะไรบ้าง")
        self.assertNotIn("next_context", result)

    def test_prior_list_does_not_create_exact_course_target(self):
        result = self._answer("IT ปี 3 เทอม 1 มีวิชาอะไรบ้าง")
        context = result.get("next_context")
        self.assertIsNotNone(context)
        self.assertIsNone(context.course_code)

    def test_unsupported_or_ambiguous_prior_turn_has_no_target_context(self):
        for question in ("IT เรียนยากไหม", "06016454 คือวิชาอะไร"):
            with self.subTest(question=question):
                result = ask(DB_PATH, question)
                self.assertNotIn("next_context", result)

    def test_context_is_frozen_structural_data_only(self):
        context = QueryContext(
            program="IT", years=(3,), semesters=(1,), course_code="06016454",
            operations=("sum_credits",),
        )
        self.assertTrue(dataclasses.is_dataclass(context))
        self.assertTrue(context.__dataclass_params__.frozen)
        self.assertEqual(
            {field.name for field in dataclasses.fields(context)},
            {"program", "plan", "years", "semesters", "category", "course_code", "operations"},
        )


if __name__ == "__main__":
    unittest.main()
