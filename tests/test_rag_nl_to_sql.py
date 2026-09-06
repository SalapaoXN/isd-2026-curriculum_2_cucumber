import unittest

from rag.structured.nl_to_sql import question_to_sql


class RagNlToSqlTest(unittest.TestCase):
    def test_fake_model_output_is_fenced_and_guarded(self):
        prompts = []

        def fake_model(prompt):
            prompts.append(prompt)
            return "```sql\nSELECT course_code FROM courses\n```"

        sql = question_to_sql(
            "วิชาในหลักสูตรมีรหัสอะไรบ้าง",
            "CREATE TABLE courses(course_id INTEGER, course_code TEXT);",
            fake_model,
        )

        self.assertEqual(sql, "SELECT course_code FROM courses LIMIT 100")
        self.assertEqual(len(prompts), 1)
        self.assertIn("course_code", prompts[0])
        self.assertIn("วิชาในหลักสูตรมีรหัสอะไรบ้าง", prompts[0])

    def test_fake_model_cannot_return_a_write_query(self):
        def fake_model(_prompt):
            return "DELETE FROM courses"

        with self.assertRaises(ValueError):
            question_to_sql("ลบข้อมูล", "CREATE TABLE courses(id INTEGER);", fake_model)


if __name__ == "__main__":
    unittest.main()
