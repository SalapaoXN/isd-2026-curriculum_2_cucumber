import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from rag.answer import EMPTY_ANSWER, answer_question
from rag.hybrid_demo import run_hybrid_demo


class RagAnswerTest(unittest.TestCase):
    def test_structured_prompt_is_grounded_in_sql_rows(self):
        prompts = []

        def answer_model(prompt):
            prompts.append(prompt)
            return "วิชา C100 มี 3 หน่วยกิต"

        answer = answer_question(
            "วิชา C100 มีกี่หน่วยกิต",
            "structured",
            structured_result={
                "columns": ["course_code", "credit_units", "source_page"],
                "rows": [("C100", 3, 12)],
            },
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, "วิชา C100 มี 3 หน่วยกิต")
        self.assertEqual(len(prompts), 1)
        self.assertIn("ผลลัพธ์จาก SQL", prompts[0])
        self.assertIn("ห้ามนับ รวม คำนวณ", prompts[0])
        self.assertIn("C100", prompts[0])
        self.assertIn("source_page", prompts[0])
        self.assertIn("12", prompts[0])

    def test_semantic_prompt_is_grounded_in_retrieved_chunks(self):
        prompts = []

        def answer_model(prompt):
            prompts.append(prompt)
            return "หลักฐานระบุเนื้อหาของวิชา C200"

        answer = answer_question(
            "วิชา C200 เรียนเกี่ยวกับอะไร",
            "semantic",
            semantic_chunks=[
                {
                    "chunk_id": "chunk-1",
                    "text": "C200 ครอบคลุมระบบฐานข้อมูล",
                    "source_page": [21],
                }
            ],
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, "หลักฐานระบุเนื้อหาของวิชา C200")
        self.assertEqual(len(prompts), 1)
        self.assertIn("ชิ้นส่วนหลักฐานที่ค้นคืนได้", prompts[0])
        self.assertIn("C200 ครอบคลุมระบบฐานข้อมูล", prompts[0])
        self.assertIn('"source_page":[21]', prompts[0])

    def test_empty_evidence_returns_exact_fallback_without_calling_model(self):
        calls = []

        def answer_model(prompt):
            calls.append(prompt)
            return "ไม่ควรถูกเรียก"

        answer = answer_question(
            "ข้อมูลที่ไม่มี",
            "structured",
            structured_result={"columns": ["course_code"], "rows": []},
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, EMPTY_ANSWER)
        self.assertEqual(calls, [])

    def test_source_page_from_chunk_provenance_is_included(self):
        prompts = []

        def answer_model(prompt):
            prompts.append(prompt)
            return "คำตอบ"

        answer_question(
            "คำถาม",
            "semantic",
            semantic_chunks=[
                {
                    "chunk_id": "chunk-2",
                    "text": "ข้อมูลอ้างอิง",
                    "provenance": [{"source_page": 33}],
                }
            ],
            answer_model_callable=answer_model,
        )

        self.assertEqual(len(prompts), 1)
        self.assertIn('"source_page":[33]', prompts[0])

    def test_combined_evidence_is_grounded_in_sql_and_chunks(self):
        prompts = []

        def answer_model(prompt):
            prompts.append(prompt)
            return "คำตอบจากหลักฐานร่วม"

        answer = answer_question(
            "วิชาฐานข้อมูลปี 1 เรียนเรื่องอะไร",
            "hybrid",
            structured_result={
                "columns": ["course_code", "source_page"],
                "rows": [("C100", 12)],
            },
            semantic_chunks=[
                {
                    "chunk_id": "chunk-1",
                    "text": "C100 ครอบคลุมฐานข้อมูล",
                    "source_page": [13],
                }
            ],
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, "คำตอบจากหลักฐานร่วม")
        self.assertEqual(len(prompts), 1)
        self.assertIn("ผลลัพธ์จาก SQL และชิ้นส่วนหลักฐาน", prompts[0])
        self.assertIn('"sql_rows":[{"course_code":"C100","source_page":12}]', prompts[0])
        self.assertIn("C100 ครอบคลุมฐานข้อมูล", prompts[0])

    def test_injected_callable_is_used(self):
        calls = []

        def answer_model(prompt):
            calls.append(prompt)
            return "คำตอบจาก callable"

        answer = answer_question(
            "คำถาม",
            "semantic",
            semantic_chunks=[{"text": "หลักฐาน"}],
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, "คำตอบจาก callable")
        self.assertEqual(len(calls), 1)

    @patch("rag.hybrid_demo.ask")
    def test_hybrid_demo_prints_grounded_final_answer(self, ask_mock):
        ask_mock.return_value = {
            "route": "semantic",
            "result": [
                {
                    "chunk_id": "chunk-1",
                    "text": "C100 เป็นวิชาพื้นฐาน",
                    "source_page": [8],
                }
            ],
        }

        output = io.StringIO()
        with redirect_stdout(output):
            response = run_hybrid_demo(
                "curriculum.db",
                "C100 คืออะไร",
                answer_model_callable=lambda prompt: "คำตอบจากหลักฐาน",
            )

        self.assertEqual(response["final_answer"], "คำตอบจากหลักฐาน")
        self.assertIn("Question: C100 คืออะไร", output.getvalue())
        self.assertIn("Final Answer: คำตอบจากหลักฐาน", output.getvalue())


if __name__ == "__main__":
    unittest.main()
