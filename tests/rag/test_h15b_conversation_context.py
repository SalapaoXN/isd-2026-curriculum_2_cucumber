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


class ConversationContextTests(unittest.TestCase):
    def test_scope_carryover_and_semester_override_requery_canonical_data(self):
        first = ask(DB_PATH, "IT ปี 3 เทอม 1 มีทั้งหมดกี่วิชา")
        context = first["next_context"]
        self.assertEqual(context.years, (3,))
        self.assertEqual(context.semesters, (1,))
        self.assertEqual(context.operations, ("count",))

        with patch("rag.qa.execute_evidence_plan", wraps=__import__(
            "rag.qa", fromlist=["execute_evidence_plan"]
        ).execute_evidence_plan) as execute:
            second = ask(DB_PATH, "แล้วเทอม 2 ล่ะ", conversation_context=context)

        self.assertIsInstance(second["result"], GroundedAnswerResult)
        self.assertEqual(second["result"].status, "answer")
        self.assertTrue(execute.called)
        self.assertTrue(
            all(claim.effective_scope.years == (3,) for claim in second["result"].claims)
        )
        self.assertTrue(
            all(
                claim.effective_scope.semesters == (2,)
                for claim in second["result"].claims
            )
        )

    def test_year_override_and_explicit_program_switch(self):
        context = QueryContext(
            program="IT", years=(3,), semesters=(1,), operations=("count",)
        )
        result = ask(DB_PATH, "DSBA แล้วปี 2 ล่ะ", conversation_context=context)
        self.assertEqual(result["result"].status, "answer")
        self.assertTrue(
            all(claim.effective_scope.program == "DSBA" for claim in result["result"].claims)
        )
        self.assertTrue(
            all(claim.effective_scope.years == (2,) for claim in result["result"].claims)
        )

    def test_exact_course_carries_to_credit_prerequisite_and_placement(self):
        first = ask(DB_PATH, "IT 06016454 คือวิชาอะไร")
        context = first["next_context"]
        self.assertEqual(context.course_code, "06016454")

        for question, expected in (
            ("แล้วกี่หน่วยกิต", "sum_credits"),
            ("มี prerequisite ไหม", "prerequisite"),
            ("วิชานี้เรียนตอนไหน", "placement"),
        ):
            with self.subTest(question=question):
                result = ask(DB_PATH, question, conversation_context=context)
                self.assertEqual(result["result"].status, "answer")
                self.assertIn(expected, [claim.operation for claim in result["result"].claims])
                self.assertTrue(result["result"].provenance)

    def test_ambiguous_course_does_not_create_exact_course_context(self):
        result = ask(DB_PATH, "06016454 คือวิชาอะไร")
        self.assertNotIn("next_context", result)

    def test_failed_turn_does_not_create_reusable_context(self):
        result = ask(DB_PATH, "IT Y5 มีวิชาอะไรบ้าง")
        self.assertNotIn("next_context", result)

    def test_context_objects_do_not_leak_into_each_other(self):
        left = QueryContext(program="IT", years=(3,), operations=("count",))
        right = QueryContext(program="DSBA", years=(2,), operations=("count",))
        left_result = ask(DB_PATH, "แล้วเทอม 1 ล่ะ", conversation_context=left)
        right_result = ask(DB_PATH, "แล้วเทอม 1 ล่ะ", conversation_context=right)
        self.assertTrue(all(c.effective_scope.program == "IT" for c in left_result["result"].claims))
        self.assertTrue(all(c.effective_scope.program == "DSBA" for c in right_result["result"].claims))


if __name__ == "__main__":
    unittest.main()
