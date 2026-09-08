import unittest

from rag.structured.nl_to_sql import question_to_sql, repair_sql


class RagNlToSqlTest(unittest.TestCase):
    def _assert_canonical_plan_guidance(self, prompt):
        for text in (
            "สหกิจ or coop -> curriculum_plans.plan_key = 'coop'",
            "ไม่สหกิจ or no_coop -> curriculum_plans.plan_key = 'no_coop'",
            "AIT/default -> curriculum_plans.plan_key = 'default'",
            "GENED -> curriculum_plans.plan_key = 'gened'",
            "Program codes belong in program or program_code, never plan_key",
            "exact plan_key equality or IN",
            "Never filter plan identity with plan_name, plan, plan_code, LIKE, NOT LIKE, or NOT IN",
        ):
            self.assertIn(text, prompt)

    def test_generation_prompt_includes_canonical_plan_guidance(self):
        prompts = []

        def fake_model(prompt):
            prompts.append(prompt)
            return "SELECT 1"

        question_to_sql("แผนสหกิจ", "curriculum_plans(plan_key)", fake_model)

        self.assertEqual(len(prompts), 1)
        self._assert_canonical_plan_guidance(prompts[0])

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

    def test_repair_returns_a_guarded_select(self):
        def fake_model(_prompt):
            return "SELECT course_code FROM courses"

        sql = repair_sql(
            "แสดงรหัสวิชา",
            "CREATE TABLE courses(course_id INTEGER, course_code TEXT);",
            "SELECT missing_code FROM courses",
            "no such column: missing_code",
            fake_model,
        )

        self.assertEqual(sql, "SELECT course_code FROM courses LIMIT 100")

    def test_repair_removes_sql_fence(self):
        def fake_model(_prompt):
            return "```sql\nSELECT course_id FROM courses\n```"

        sql = repair_sql(
            "แสดงรหัสวิชา",
            "CREATE TABLE courses(course_id INTEGER, course_code TEXT);",
            "SELECT missing_code FROM courses",
            "no such column: missing_code",
            fake_model,
        )

        self.assertEqual(sql, "SELECT course_id FROM courses LIMIT 100")

    def test_repair_rejects_a_write_query(self):
        def fake_model(_prompt):
            return "DELETE FROM courses"

        with self.assertRaises(ValueError):
            repair_sql(
                "แสดงรหัสวิชา",
                "CREATE TABLE courses(course_id INTEGER, course_code TEXT);",
                "SELECT missing_code FROM courses",
                "no such column: missing_code",
                fake_model,
            )

    def test_repair_prompt_contains_failed_sql_and_error(self):
        prompts = []

        def fake_model(prompt):
            prompts.append(prompt)
            return "SELECT course_code FROM courses"

        repair_sql(
            "แสดงรหัสวิชา",
            "CREATE TABLE courses(course_id INTEGER, course_code TEXT);",
            "SELECT missing_code FROM courses",
            "no such column: missing_code",
            fake_model,
        )

        self.assertEqual(len(prompts), 1)
        self.assertIn("SELECT missing_code FROM courses", prompts[0])
        self.assertIn("no such column: missing_code", prompts[0])

    def test_repair_prompt_includes_canonical_plan_guidance(self):
        prompts = []

        def fake_model(prompt):
            prompts.append(prompt)
            return "SELECT 1"

        repair_sql(
            "แผนไม่สหกิจ",
            "curriculum_plans(plan_key, plan_name, plan, plan_code)",
            "SELECT * FROM curriculum_plans WHERE plan_name LIKE '%ไม่สหกิจ%'",
            "unsafe plan filter",
            fake_model,
        )

        self.assertEqual(len(prompts), 1)
        self._assert_canonical_plan_guidance(prompts[0])


if __name__ == "__main__":
    unittest.main()
