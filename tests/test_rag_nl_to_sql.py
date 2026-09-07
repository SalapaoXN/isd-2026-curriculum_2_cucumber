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

    def test_course_name_question_instructs_model_to_join_courses(self):
        prompts = []

        def fake_model(prompt):
            prompts.append(prompt)
            return (
                "SELECT DISTINCT v.course_code, c.name_th "
                "FROM v_plan_courses AS v "
                "JOIN courses AS c ON c.course_id = v.course_id "
                "WHERE v.program = 'IT' AND v.year = 1 AND v.semester = 1"
            )

        sql = question_to_sql(
            "IT ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
            "CREATE TABLE courses(course_id INTEGER, name_th TEXT, name_en TEXT);",
            fake_model,
        )

        self.assertEqual(
            sql,
            "SELECT DISTINCT v.course_code, c.name_th FROM v_plan_courses AS v "
            "JOIN courses AS c ON c.course_id = v.course_id "
            "WHERE v.program = 'IT' AND v.year = 1 AND v.semester = 1 LIMIT 100",
        )
        self.assertIn("v_plan_courses does NOT expose course-name columns", prompts[0])
        self.assertIn(
            "JOIN courses ON courses.course_id = v_plan_courses.course_id",
            prompts[0],
        )
        self.assertIn("courses.name_th or courses.name_en", prompts[0])


if __name__ == "__main__":
    unittest.main()
