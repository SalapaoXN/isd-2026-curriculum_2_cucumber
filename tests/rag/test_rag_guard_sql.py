import unittest

from rag.structured.guard_sql import (
    extract_column_predicate_literals,
    guard_sql,
)


class RagGuardSqlTest(unittest.TestCase):
    ALLOWED_RELATIONS = {
        "courses",
        "v_plan_courses",
        "v_semester_credits",
        "v_prerequisite_edges",
        "curriculum_plans",
        "programs",
    }

    def test_allows_read_queries_and_bounds_limits(self):
        self.assertEqual(
            guard_sql("SELECT course_code FROM courses"),
            "SELECT course_code FROM courses LIMIT 100",
        )
        self.assertEqual(
            guard_sql("WITH selected AS (SELECT 1) SELECT * FROM selected"),
            "WITH selected AS (SELECT 1) SELECT * FROM selected LIMIT 100",
        )
        self.assertEqual(
            guard_sql("SELECT * FROM courses LIMIT 5"),
            "SELECT * FROM courses LIMIT 5",
        )
        self.assertEqual(
            guard_sql("SELECT * FROM courses LIMIT 500"),
            "SELECT * FROM courses LIMIT 100",
        )
        self.assertEqual(
            guard_sql("SELECT * FROM courses;"),
            "SELECT * FROM courses LIMIT 100;",
        )

    def test_rejects_writes_and_multiple_statements(self):
        for sql in (
            "INSERT INTO courses VALUES (1)",
            "UPDATE courses SET course_code = 'x'",
            "DELETE FROM courses",
            "DROP TABLE courses",
            "ALTER TABLE courses ADD COLUMN x TEXT",
            "PRAGMA foreign_keys",
            "ATTACH DATABASE 'other.db' AS other",
            "SELECT 1; SELECT 2",
        ):
            with self.subTest(sql=sql):
                with self.assertRaises(ValueError):
                    guard_sql(sql)

    def test_ignores_keywords_in_literals_and_comments(self):
        sql = "SELECT 'DROP; UPDATE' AS value /* ATTACH */ -- DELETE\n"
        self.assertEqual(
            guard_sql(sql),
            "SELECT 'DROP; UPDATE' AS value LIMIT 100 /* ATTACH */ -- DELETE\n",
        )

    def test_relation_allowlist_accepts_allowed_tables_and_joins(self):
        self.assertEqual(
            guard_sql(
                "SELECT c.course_code FROM courses AS c",
                allowed_relations=self.ALLOWED_RELATIONS,
            ),
            "SELECT c.course_code FROM courses AS c LIMIT 100",
        )
        self.assertEqual(
            guard_sql(
                "SELECT c.course_code FROM courses c JOIN v_plan_courses p "
                "ON p.course_id = c.course_id",
                allowed_relations=self.ALLOWED_RELATIONS,
            ),
            "SELECT c.course_code FROM courses c JOIN v_plan_courses p "
            "ON p.course_id = c.course_id LIMIT 100",
        )

    def test_relation_allowlist_rejects_disallowed_nested_and_cte_relations(self):
        for sql in (
            "SELECT * FROM secret_table",
            "SELECT * FROM courses WHERE course_id IN "
            "(SELECT course_id FROM secret_table)",
            "WITH selected AS (SELECT * FROM secret_table) "
            "SELECT * FROM selected",
        ):
            with self.subTest(sql=sql):
                with self.assertRaises(ValueError):
                    guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS)

    def test_relation_allowlist_allows_cte_when_physical_sources_are_allowed(self):
        sql = (
            "WITH selected AS (SELECT course_id FROM courses) "
            "SELECT * FROM selected"
        )
        self.assertEqual(
            guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS),
            f"{sql} LIMIT 100",
        )

    def test_relation_allowlist_accepts_allowed_derived_table_in_from(self):
        sql = (
            "SELECT x.course_id FROM (SELECT course_id FROM courses) AS x "
            "LIMIT 100"
        )
        self.assertEqual(
            guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS),
            sql,
        )

    def test_relation_allowlist_accepts_allowed_derived_table_in_join(self):
        sql = (
            "SELECT c.course_id FROM courses AS c "
            "JOIN (SELECT course_id FROM courses) AS x "
            "ON x.course_id = c.course_id LIMIT 100"
        )
        self.assertEqual(
            guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS),
            sql,
        )

    def test_relation_allowlist_rejects_disallowed_derived_table_relation(self):
        sql = (
            "SELECT x.course_id FROM (SELECT course_id FROM secret_table) AS x "
            "LIMIT 100"
        )
        with self.assertRaises(ValueError):
            guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS)

    def test_relation_allowlist_accepts_cte_column_list(self):
        sql = (
            "WITH selected(course_id) AS (SELECT course_id FROM courses) "
            "SELECT course_id FROM selected LIMIT 100"
        )
        self.assertEqual(
            guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS),
            sql,
        )

    def test_relation_allowlist_rejects_disallowed_cte_column_list_relation(self):
        sql = (
            "WITH selected(course_id) AS (SELECT course_id FROM secret_table) "
            "SELECT course_id FROM selected LIMIT 100"
        )
        with self.assertRaises(ValueError):
            guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS)

    def test_relation_allowlist_accepts_multiple_ctes_with_column_lists(self):
        sql = (
            "WITH first_set(course_id) AS (SELECT course_id FROM courses), "
            "second_set(course_id) AS (SELECT course_id FROM v_plan_courses) "
            "SELECT first_set.course_id FROM first_set "
            "JOIN second_set ON second_set.course_id = first_set.course_id "
            "LIMIT 100"
        )
        self.assertEqual(
            guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS),
            sql,
        )

    def test_relation_allowlist_accepts_chained_ctes_and_set_operations(self):
        sql = (
            "WITH base AS (SELECT course_code FROM v_plan_courses), "
            "filtered AS (SELECT course_code FROM base), "
            "other AS (SELECT course_code FROM courses) "
            "SELECT course_code, 'base' AS source FROM filtered "
            "EXCEPT SELECT course_code, 'other' FROM other"
        )
        self.assertEqual(
            guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS),
            f"{sql} LIMIT 100",
        )

    def test_relation_allowlist_does_not_leak_nested_cte_names_to_outer_query(self):
        sql = (
            "WITH outer_set AS ("
            "WITH inner_only AS (SELECT course_id FROM courses) "
            "SELECT course_id FROM inner_only"
            ") SELECT course_id FROM inner_only"
        )
        with self.assertRaisesRegex(ValueError, "relation is not allowed: inner_only"):
            guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS)

    def test_relation_allowlist_does_not_allow_cte_name_to_hide_its_own_physical_source(self):
        sql = (
            "WITH secret_table AS (SELECT course_code FROM secret_table) "
            "SELECT course_code FROM secret_table"
        )
        with self.assertRaisesRegex(ValueError, "relation is not allowed: secret_table"):
            guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS)

    def test_cte_alias_is_not_added_to_the_physical_allowlist(self):
        with self.assertRaisesRegex(ValueError, "relation is not allowed: local_set"):
            guard_sql(
                "SELECT course_code FROM local_set",
                allowed_relations=self.ALLOWED_RELATIONS,
            )

    def test_relation_allowlist_rejects_unauthorized_relation_after_valid_cte(self):
        sql = (
            "WITH selected AS (SELECT course_id FROM courses) "
            "SELECT course_id FROM selected UNION "
            "SELECT course_id FROM secret_table"
        )
        with self.assertRaisesRegex(ValueError, "relation is not allowed: secret_table"):
            guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS)

    def test_relation_allowlist_keeps_recursive_ctes_unsupported(self):
        sql = (
            "WITH RECURSIVE seq(n) AS (SELECT 1 UNION ALL "
            "SELECT n + 1 FROM seq WHERE n < 3) SELECT n FROM seq"
        )
        with self.assertRaisesRegex(ValueError, "WITH RECURSIVE is not supported"):
            guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS)

    def test_relation_allowlist_rejects_malformed_cte_syntax(self):
        for sql in (
            "WITH selected AS SELECT 1",
            "WITH selected(course_id) (SELECT course_id FROM courses) SELECT 1",
            "WITH selected(course_id,) AS (SELECT course_id FROM courses) "
            "SELECT course_id FROM selected",
            "WITH selected AS (SELECT course_id FROM courses) "
            "other AS (SELECT course_id FROM courses) SELECT course_id FROM selected",
        ):
            with self.subTest(sql=sql):
                with self.assertRaises(ValueError):
                    guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS)

    def test_relation_allowlist_accepts_nested_cte_body(self):
        sql = (
            "WITH selected AS ("
            "WITH inner_set AS (SELECT course_id FROM courses) "
            "SELECT course_id FROM inner_set"
            ") SELECT course_id FROM selected LIMIT 100"
        )
        self.assertEqual(
            guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS),
            sql,
        )

    def test_relation_allowlist_rejects_ambiguous_parenthesized_relation_source(self):
        with self.assertRaises(ValueError):
            guard_sql(
                "SELECT * FROM (courses) AS selected LIMIT 100",
                allowed_relations=self.ALLOWED_RELATIONS,
            )

    def test_relation_allowlist_rejects_schema_qualified_relations(self):
        with self.assertRaises(ValueError):
            guard_sql(
                "SELECT * FROM main.courses",
                allowed_relations=self.ALLOWED_RELATIONS,
            )

    def test_relation_allowlist_ignores_from_text_inside_sql_strings(self):
        sql = "SELECT 'FROM secret_table' AS text FROM courses"
        self.assertEqual(
            guard_sql(sql, allowed_relations=self.ALLOWED_RELATIONS),
            f"{sql} LIMIT 100",
        )

    def test_extracts_literals_only_from_qualified_course_code_predicates(self):
        sql = (
            "SELECT c.course_code FROM courses AS c "
            "WHERE c.course_code = '06026200' "
            "OR c.course_code_normalized IN ('MATH102', 'MATH112') "
            "AND c.name_en LIKE '%Calculus 2%' "
            "AND program = 'IT' AND plan_key = 'coop'"
        )

        self.assertEqual(
            extract_column_predicate_literals(
                sql, {"course_code", "course_code_normalized"}
            ),
            ("06026200", "MATH102", "MATH112"),
        )


if __name__ == "__main__":
    unittest.main()
