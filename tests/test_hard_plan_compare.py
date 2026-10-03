import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from backend.hard_plan_compare import (
    compare_curriculum_editions,
    compare_plan_course_sets,
)


class HardPlanCompareTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "curriculum.db"
        schema_path = Path(__file__).parents[1] / "rag" / "structured" / "schema.sql"
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executescript(schema_path.read_text(encoding="utf-8"))
            self._seed_fixture(connection)
            connection.commit()

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _seed_fixture(connection):
        connection.executemany(
            "INSERT INTO catalogs(catalog_id, catalog_key) VALUES (?, ?)",
            [(1, "left"), (2, "right")],
        )
        connection.executemany(
            "INSERT INTO programs VALUES (?, ?, ?, ?)",
            [(1, 1, "DSBA", "dsba"), (2, 2, "DSBA", "dsba")],
        )
        connection.executemany(
            """INSERT INTO courses
               (course_id, catalog_id, course_code, course_code_normalized,
                name_th, name_en)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (1, 1, "00000001", "00000001", "ชื่อซ้ำ", "Shared name"),
                (2, 1, "00000002", "00000002", "วิชาร่วม", "Shared"),
                (3, 2, "00000002", "00000002", "วิชาร่วม", "Shared"),
                (4, 2, "00000003", "00000003", "ชื่อซ้ำ", "Shared name"),
            ],
        )
        connection.executemany(
            """INSERT INTO curriculum_plans
               (plan_id, catalog_id, program_id, program_code, plan_key)
               VALUES (?, ?, ?, ?, ?)""",
            [(1, 1, 1, "DSBA", "coop"), (2, 2, 2, "DSBA", "no_coop")],
        )
        connection.executemany(
            """INSERT INTO plan_placements
               (placement_id, plan_id, course_id, year_number, semester_number)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (11, 1, 1, 1, 1),
                (12, 1, 1, 1, 2),  # repeated placement is one logical course
                (13, 1, 2, 1, 1),
                (21, 2, 3, 1, 1),  # separate physical course, same logical code
                (22, 2, 4, 1, 1),
            ],
        )
        connection.executemany(
            """INSERT INTO provenance
               (provenance_id, source_document_key, program, source_filename,
                source_page, document_page, document_category)
               VALUES (?, ?, ?, ?, ?, ?, 'plan')""",
            [
                (101, "left-plan", "DSBA", "left.pdf", 1, 1),
                (102, "right-plan", "DSBA", "right.pdf", 2, 2),
                (111, "left-only-a", "DSBA", "left.pdf", 3, 2),
                (112, "left-only-b", "DSBA", "left.pdf", 4, None),
                (113, "left-course", "DSBA", "left.pdf", 5, 4),
                (121, "left-shared", "DSBA", "left.pdf", 6, 5),
                (122, "right-shared", "DSBA", "right.pdf", 7, 6),
                (131, "right-only-a", "DSBA", "right.pdf", 8, 7),
                (132, "right-only-b", "DSBA", "right.pdf", 9, None),
            ],
        )
        connection.executemany(
            "INSERT INTO curriculum_plan_provenance VALUES (?, ?)",
            [(1, 101), (2, 102)],
        )
        connection.executemany(
            "INSERT INTO plan_placement_provenance VALUES (?, ?)",
            [(11, 111), (12, 112), (13, 121), (21, 122), (22, 131)],
        )
        connection.executemany(
            "INSERT INTO course_provenance(course_id, provenance_id) VALUES (?, ?)",
            [(1, 113), (2, 121), (3, 122), (4, 132)],
        )

    def test_explicit_plan_comparison_uses_normalized_code_and_deduplicates(self):
        result = compare_plan_course_sets(self.db_path, "DSBA", "coop", "no_coop")

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["program"], "DSBA")
        self.assertEqual(result["left_plan"], "coop")
        self.assertEqual(result["right_plan"], "no_coop")
        self.assertEqual(result["left_course_count"], 2)
        self.assertEqual(result["right_course_count"], 2)
        self.assertEqual(result["shared_course_count"], 1)
        self.assertEqual(
            [course["course_code_normalized"] for course in result["left_only_courses"]],
            ["00000001"],
        )
        self.assertEqual(
            [course["course_code_normalized"] for course in result["right_only_courses"]],
            ["00000003"],
        )
        self.assertEqual(result["shared_courses"][0]["course_code"], "00000002")
        self.assertEqual(len(result["left_only_courses"][0]["placement_ids"]), 2)

    def test_names_do_not_collapse_different_codes_and_provenance_is_retained(self):
        result = compare_plan_course_sets(self.db_path, "DSBA", "coop", "no_coop")
        left_only = result["left_only_courses"][0]
        right_only = result["right_only_courses"][0]

        self.assertEqual(left_only["name_en"], "Shared name")
        self.assertEqual(right_only["name_en"], "Shared name")
        self.assertEqual(
            {item["provenance_id"] for item in left_only["provenance"]},
            {101, 111, 112, 113},
        )
        self.assertEqual(
            {item["provenance_id"] for item in right_only["provenance"]},
            {102, 131, 132},
        )
        missing_document_page = next(
            item for item in left_only["provenance"] if item["provenance_id"] == 112
        )
        self.assertEqual(missing_document_page["source_filename"], "left.pdf")
        self.assertEqual(missing_document_page["source_page"], 4)
        self.assertIsNone(missing_document_page["document_page"])

    def test_alternative_members_are_canonical_plan_course_candidates(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """INSERT INTO alternative_course_groups
                   (alternative_group_id, catalog_id, plan_id, group_key)
                   VALUES (31, 1, 1, 'choose-one')"""
            )
            connection.executemany(
                """INSERT INTO alternative_course_group_members
                   (alternative_group_member_id, alternative_group_id,
                    course_id, member_order)
                   VALUES (?, 31, ?, ?)""",
                [(311, 1, 1), (312, 2, 2)],
            )
            connection.execute(
                """INSERT INTO plan_placements
                   (placement_id, plan_id, alternative_group_id)
                   VALUES (14, 1, 31)"""
            )
            connection.execute(
                """INSERT INTO provenance
                   (provenance_id, source_document_key, program, source_filename,
                    source_page, document_page, document_category)
                   VALUES (114, 'left-alternative', 'DSBA', 'left.pdf', 10, NULL, 'plan')"""
            )
            connection.execute(
                "INSERT INTO plan_placement_provenance VALUES (14, 114)"
            )
            connection.execute(
                "INSERT INTO alternative_group_provenance VALUES (31, 114)"
            )
            connection.executemany(
                "INSERT INTO alternative_group_member_provenance VALUES (?, ?)",
                [(311, 114), (312, 114)],
            )
            connection.commit()

        result = compare_plan_course_sets(self.db_path, "DSBA", "coop", "no_coop")

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["left_course_count"], 2)
        left_only = result["left_only_courses"][0]
        self.assertEqual(left_only["course_code_normalized"], "00000001")
        self.assertIn(14, left_only["placement_ids"])
        self.assertIn(114, {item["provenance_id"] for item in left_only["provenance"]})

    def test_missing_or_ambiguous_scope_returns_controlled_status(self):
        missing = compare_plan_course_sets(self.db_path, "DSBA", "coop", "default")
        self.assertEqual(missing["status"], "plan_not_found")

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("INSERT INTO catalogs(catalog_id) VALUES (3)")
            connection.execute(
                "INSERT INTO programs VALUES (3, 3, 'DSBA', 'dsba')"
            )
            connection.execute(
                """INSERT INTO curriculum_plans
                   (plan_id, catalog_id, program_id, program_code, plan_key)
                   VALUES (3, 3, 3, 'DSBA', 'no_coop')"""
            )
            connection.commit()
        ambiguous = compare_plan_course_sets(self.db_path, "DSBA", "coop", "no_coop")
        self.assertEqual(ambiguous["status"], "ambiguous_plan")
        selected = compare_plan_course_sets(
            self.db_path, "DSBA", "coop", "no_coop", catalog_key="left"
        )
        self.assertEqual(selected["status"], "plan_not_found")

    def test_missing_membership_provenance_fails_closed(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("DELETE FROM plan_placement_provenance WHERE placement_id IN (11, 12)")
            connection.execute("DELETE FROM course_provenance WHERE course_id = 1")
            connection.commit()

        result = compare_plan_course_sets(self.db_path, "DSBA", "coop", "no_coop")

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["left_only_courses"], [])
        self.assertTrue(result["limitations"])

    def test_database_is_opened_read_only(self):
        original_connect = sqlite3.connect
        seen = []

        def capture_connect(database, *args, **kwargs):
            seen.append((str(database), kwargs.get("uri")))
            return original_connect(database, *args, **kwargs)

        with patch("backend.hard_plan_compare.sqlite3.connect", side_effect=capture_connect):
            result = compare_plan_course_sets(self.db_path, "DSBA", "coop", "no_coop")

        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(seen), 1)
        self.assertIn("mode=ro", seen[0][0])
        self.assertTrue(seen[0][1])

    def test_current_dsba_plans_have_equal_logical_course_sets(self):
        db_path = Path("cucumber_outputs/runtime/curriculum.db")
        if not db_path.exists():
            self.skipTest("runtime curriculum DB is not present")

        result = compare_plan_course_sets(
            db_path, "DSBA", "coop", "no_coop", catalog_key="dsba-2565"
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["catalog_key"], "dsba-2565")
        self.assertEqual(result["left_course_count"], 83)
        self.assertEqual(result["right_course_count"], 83)
        self.assertEqual(result["shared_course_count"], 83)
        self.assertEqual(result["left_only_courses"], [])
        self.assertEqual(result["right_only_courses"], [])
        self.assertTrue(result["left_plan_evidence"])
        self.assertTrue(result["right_plan_evidence"])

    def test_both_dsba_editions_resolve_coop_and_no_coop_independently(self):
        db_path = Path("cucumber_outputs/runtime/curriculum.db")
        if not db_path.exists():
            self.skipTest("runtime curriculum DB is not present")

        for catalog_key in ("dsba-2560", "dsba-2565"):
            with self.subTest(catalog_key=catalog_key):
                result = compare_plan_course_sets(
                    db_path, "DSBA", "coop", "no_coop", catalog_key=catalog_key
                )
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["catalog_key"], catalog_key)

    def test_old_new_comparison_returns_four_deterministic_code_categories(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "UPDATE catalogs SET academic_year='2560' WHERE catalog_id=1"
            )
            connection.execute(
                "UPDATE catalogs SET academic_year='2565' WHERE catalog_id=2"
            )
            connection.commit()

        first = compare_curriculum_editions(self.db_path, "DSBA")
        second = compare_curriculum_editions(self.db_path, "DSBA")

        self.assertEqual(first, second)
        self.assertEqual(first["status"], "complete")
        categories = first["categories"]
        self.assertEqual(
            {item["course_code_normalized"] for item in categories["shared_same_code"]},
            {"00000002"},
        )
        self.assertEqual(
            {item["course_code_normalized"] for item in categories["old_only_by_code"]},
            {"00000001"},
        )
        self.assertEqual(
            {item["course_code_normalized"] for item in categories["new_only_by_code"]},
            {"00000003"},
        )
        candidate = categories["same_name_changed_code_candidates"][0]
        self.assertEqual(candidate["older"]["course_code"], "00000001")
        self.assertEqual(candidate["newer"]["course_code"], "00000003")
        self.assertFalse(candidate["equivalence_proven"])
        self.assertTrue(candidate["older"]["provenance"])
        self.assertTrue(candidate["newer"]["provenance"])
        self.assertEqual(
            categories["old_only_by_code"][0]["candidate_code_changes"],
            [{"course_code_normalized": "00000003", "equivalence_proven": False}],
        )
        self.assertEqual(
            categories["new_only_by_code"][0]["candidate_code_changes"],
            [{"course_code_normalized": "00000001", "equivalence_proven": False}],
        )

    def test_old_new_comparison_can_preserve_a_requested_plan_scope(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "UPDATE catalogs SET academic_year='2560' WHERE catalog_id=1"
            )
            connection.execute(
                "UPDATE catalogs SET academic_year='2565' WHERE catalog_id=2"
            )
            connection.execute(
                "INSERT INTO curriculum_plans (plan_id,catalog_id,program_id,program_code,plan_key) "
                "VALUES (3,2,2,'DSBA','coop')"
            )
            connection.execute(
                "INSERT INTO plan_placements (placement_id,plan_id,course_id,year_number,semester_number) "
                "VALUES (23,3,3,1,1)"
            )
            connection.execute(
                "INSERT INTO plan_placement_provenance VALUES (23,122)"
            )
            connection.commit()

        result = compare_curriculum_editions(self.db_path, "DSBA", plan="coop")

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["plan"], "coop")
        for category in ("shared_same_code", "old_only_by_code", "new_only_by_code"):
            for bucket in result["categories"][category]:
                courses = bucket.get("courses", []) or bucket.get("older", []) + bucket.get("newer", [])
                for course in courses:
                    self.assertEqual(course["plans"], ["coop"])

    def test_old_new_comparison_fails_closed_when_plan_is_missing_from_an_edition(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "UPDATE catalogs SET academic_year='2560' WHERE catalog_id=1"
            )
            connection.execute(
                "UPDATE catalogs SET academic_year='2565' WHERE catalog_id=2"
            )
            connection.execute(
                "INSERT INTO curriculum_plans (plan_id,catalog_id,program_id,program_code,plan_key) "
                "VALUES (3,2,2,'DSBA','coop')"
            )
            connection.execute(
                "INSERT INTO plan_placements (placement_id,plan_id,course_id,year_number,semester_number) "
                "VALUES (23,3,3,1,1)"
            )
            connection.execute(
                "INSERT INTO plan_placement_provenance VALUES (23,122)"
            )
            connection.commit()

        result = compare_curriculum_editions(self.db_path, "DSBA", plan="gened")

        self.assertEqual(result["status"], "plan_unavailable")
        self.assertIn("not present in both", result["limitations"][0])

    def test_old_new_comparison_keeps_unmatched_codes_and_placeholders_separate(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "UPDATE catalogs SET academic_year='2560' WHERE catalog_id=1"
            )
            connection.execute(
                "UPDATE catalogs SET academic_year='2565' WHERE catalog_id=2"
            )
            connection.executemany(
                """INSERT INTO courses
                   (course_id,catalog_id,course_code,course_code_normalized,name_th,name_en)
                   VALUES (?,?,?,?,?,?)""",
                [
                    (5, 1, "00000004", "00000004", "วิชาเก่าที่ไม่ตรง", "Old unmatched"),
                    (6, 2, "00000005", "00000005", "วิชาใหม่ที่ไม่ตรง", "New unmatched"),
                    (7, 1, "060261xx", "060261XX", "กลุ่มวิชา", "Course group"),
                ],
            )
            connection.executemany(
                """INSERT INTO plan_placements
                   (placement_id,plan_id,course_id,year_number,semester_number)
                   VALUES (?,?,?,?,?)""",
                [(14, 1, 5, 2, 1), (24, 2, 6, 2, 1), (15, 1, 7, 2, 1)],
            )
            connection.executemany(
                """INSERT INTO provenance
                   (provenance_id,source_document_key,program,source_filename,
                    source_page,document_page,document_category)
                   VALUES (?,?,?,?,?,?, 'plan')""",
                [
                    (133, "old-unmatched", "DSBA", "left.pdf", 10, 10),
                    (134, "new-unmatched", "DSBA", "right.pdf", 11, 11),
                    (135, "placeholder", "DSBA", "left.pdf", 12, 12),
                ],
            )
            connection.executemany(
                "INSERT INTO plan_placement_provenance VALUES (?,?)",
                [(14, 133), (24, 134), (15, 135)],
            )
            connection.executemany(
                "INSERT INTO course_provenance(course_id,provenance_id) VALUES (?,?)",
                [(5, 133), (6, 134), (7, 135)],
            )
            connection.commit()

        result = compare_curriculum_editions(self.db_path, "DSBA")
        categories = result["categories"]
        old_unmatched = next(
            item for item in categories["old_only_by_code"]
            if item["course_code_normalized"] == "00000004"
        )
        new_unmatched = next(
            item for item in categories["new_only_by_code"]
            if item["course_code_normalized"] == "00000005"
        )

        self.assertEqual(old_unmatched["candidate_code_changes"], [])
        self.assertEqual(new_unmatched["candidate_code_changes"], [])
        self.assertEqual(
            [item["course_code"] for item in categories["unresolved_non_concrete"]],
            ["060261XX"],
        )
        self.assertNotIn(
            "060261XX",
            {item["course_code_normalized"] for item in categories["old_only_by_code"]},
        )
        source_files = {item["source_filename"] for item in result["provenance"]}
        self.assertIn("left.pdf", source_files)
        self.assertIn("right.pdf", source_files)

    def test_old_new_comparison_requires_exactly_two_distinct_editions(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "INSERT INTO catalogs(catalog_id,catalog_key,academic_year) VALUES (3,'third','2570')"
            )
            connection.execute("INSERT INTO programs VALUES (3,3,'DSBA','dsba')")
            connection.commit()

        result = compare_curriculum_editions(self.db_path, "DSBA")

        self.assertEqual(result["status"], "ambiguous_edition")
        self.assertTrue(result["limitations"])


if __name__ == "__main__":
    unittest.main()
