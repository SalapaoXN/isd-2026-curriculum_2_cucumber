"""Provider-free, side-local concrete course-set comparison regressions."""

from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from rag.semantic.executor import _side_course_set, execute_comparison
from rag.semantic.pipeline import semantic_answer
from rag.semantic.schema import (
    ComparisonSpec, ResolvedIntent, ResolvedOperand, ResolvedScope, SemanticIntent,
)


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"
PROVENANCE = ({"source_filename": "fixture.pdf", "source_page": 1},)


class CourseSetOverlapTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.db = Path(temp.name) / "comparison.db"
        with closing(sqlite3.connect(self.db)) as connection:
            connection.executescript("""
                CREATE TABLE catalogs (catalog_id INTEGER, catalog_key TEXT);
                CREATE TABLE programs (program_id INTEGER, program_code TEXT);
                CREATE TABLE curriculum_plans (
                    plan_id INTEGER, program_id INTEGER, catalog_id INTEGER, plan_key TEXT);
                CREATE TABLE courses (course_id INTEGER, course_code TEXT, name_th TEXT);
                CREATE TABLE plan_placements (
                    placement_id INTEGER PRIMARY KEY, plan_id INTEGER, course_id INTEGER,
                    year_number INTEGER, semester_number INTEGER, flexible_year_semester_raw TEXT);
                INSERT INTO catalogs VALUES (1, 'alpha-2601'), (2, 'beta-2602');
                INSERT INTO programs VALUES (1, 'ALPHA'), (2, 'BETA');
                INSERT INTO curriculum_plans VALUES
                    (1, 1, 1, 'coop'), (2, 2, 2, 'default'), (3, 1, 1, 'no_coop');
            """)
        self.left = ResolvedOperand(unresolved=False, scope=ResolvedScope(
            program="ALPHA", catalog_key="alpha-2601", years=(1,)))
        self.right = ResolvedOperand(unresolved=False, scope=ResolvedScope(
            program="BETA", catalog_key="beta-2602", years=(1,)))

    def add(self, course_id, code, plan, year=1, semester=1, flexible=None):
        with closing(sqlite3.connect(self.db)) as connection:
            if not connection.execute(
                "SELECT 1 FROM courses WHERE course_id = ?", (course_id,)
            ).fetchone():
                connection.execute("INSERT INTO courses VALUES (?, ?, ?)",
                                   (course_id, code, "Same grounded title"))
            connection.execute(
                "INSERT INTO plan_placements VALUES (NULL, ?, ?, ?, ?, ?)",
                (plan, course_id, year, semester, flexible),
            )
            connection.commit()

    def compare(self, operation="overlap", *, left=None, right=None, provenance=PROVENANCE):
        intent = SemanticIntent(task="compare", comparison=ComparisonSpec(
            measure="course_count", operation=operation))
        resolved = ResolvedIntent(intent=intent, comparison_sides=(
            left or self.left, right or self.right))
        with patch("rag.semantic.executor._hydrated_provenance", return_value=provenance) as hydrate:
            result = execute_comparison(self.db, resolved)
        return result, hydrate

    def assert_codes(self, result, expected):
        self.assertEqual(result.status, "answer", result.missing_information)
        self.assertEqual(re.findall(r"\b[0-9]{8}\b", "\n".join(result.summary_facts)), expected)

    def test_year_scope_precedes_dedup_and_excludes_shared_other_years(self):
        self.add(11, "10000001", 1, year=2)
        self.add(11, "10000001", 1, year=1)
        self.add(21, "10000001", 2)
        self.add(12, "10000002", 1, year=2)
        self.add(22, "10000002", 2, year=2)
        result, hydrate = self.compare()
        self.assert_codes(result, ["10000001"])
        hydrate.assert_called_once_with(self.db, (11, 21))

    def test_asymmetric_year_and_semester_scopes(self):
        self.add(11, "10000001", 1, year=1, semester=2)
        self.add(21, "10000001", 2, year=2, semester=1)
        self.add(12, "10000002", 1, year=1, semester=1)
        self.add(22, "10000002", 2, year=2, semester=1)
        self.add(13, "10000003", 1, year=1, semester=2)
        self.add(23, "10000003", 2, year=1, semester=2)
        left = replace(self.left, scope=replace(self.left.scope, semesters=(2,)))
        right = replace(self.right, scope=replace(self.right.scope, years=(2,), semesters=(1,)))
        result, _ = self.compare(left=left, right=right)
        self.assert_codes(result, ["10000001"])

    def test_flexible_temporal_scope_must_match_one_choice(self):
        self.add(11, "10000001", 1, year=None, semester=None, flexible="1/2, 2/1")
        self.add(21, "10000001", 2)
        self.add(12, "10000002", 1)
        self.add(22, "10000002", 2)
        for years, semesters, expected in (
            ((1,), (1,), ["10000002"]),
            ((1,), (2,), ["10000001"]),
            ((1,), (), ["10000001", "10000002"]),
            ((), (2,), ["10000001"]),
        ):
            with self.subTest(years=years, semesters=semesters):
                left = replace(self.left, scope=replace(
                    self.left.scope, years=years, semesters=semesters))
                result, _ = self.compare(left=left)
                self.assert_codes(result, expected)

    def test_masks_do_not_block_overlap_or_become_shared_identities(self):
        self.add(11, "10000001", 1)
        self.add(21, "10000001", 2)
        for index, mask in enumerate(("06026xxx", "9064xxxx", "xxxxxxxx", "060464XX"), 30):
            self.add(index, mask, 1)
            self.add(index + 10, mask, 2)
        result, hydrate = self.compare()
        self.assert_codes(result, ["10000001"])
        hydrate.assert_called_once_with(self.db, (11, 21))

    def test_repeated_placements_and_plans_deduplicate_but_keep_catalog_ids(self):
        self.add(11, "10000001", 1)
        self.add(11, "10000001", 1, semester=2)
        self.add(11, "10000001", 3)
        self.add(21, "10000001", 2)
        self.add(21, "10000001", 2)
        result, hydrate = self.compare()
        self.assert_codes(result, ["10000001"])
        hydrate.assert_called_once_with(self.db, (11, 21))

    def test_optional_plan_filter_remains_side_local(self):
        self.add(11, "10000001", 1)
        self.add(12, "10000002", 3)
        self.add(21, "10000001", 2)
        self.add(22, "10000002", 2)
        left = replace(self.left, scope=replace(self.left.scope, plan="no_coop"))
        result, _ = self.compare(left=left)
        self.assert_codes(result, ["10000002"])

    def test_same_title_different_code_is_verified_empty_overlap(self):
        self.add(11, "10000001", 1)
        self.add(21, "10000002", 2)
        self.add(12, "9064xxxx", 1)
        self.add(22, "9064xxxx", 2)
        result, hydrate = self.compare()
        self.assert_codes(result, [])
        self.assertIn("0 วิชา", result.summary_facts[0])
        self.assertEqual(result.provenance, PROVENANCE)
        hydrate.assert_called_once_with(self.db, (11, 21))

    def test_missing_provenance_fails_closed(self):
        self.add(11, "10000001", 1)
        self.add(21, "10000001", 2)
        for provenance in (None, ()):
            with self.subTest(provenance=provenance):
                result, _ = self.compare(provenance=provenance)
                self.assertEqual(result.status, "missing_data")
                self.assertEqual(result.failure_category, "DATA_ERROR")

    def test_difference_fails_closed_for_in_scope_mask_on_either_side(self):
        self.add(11, "10000001", 1)
        self.add(21, "10000002", 2)
        self.add(22, "9064xxxx", 2)
        for left, right in ((self.left, self.right), (self.right, self.left)):
            with self.subTest(left=left.scope.program):
                result, hydrate = self.compare("set_difference", left=left, right=right)
                self.assertEqual(result.status, "missing_data")
                self.assertIn("course set contains masked placeholder identities", result.missing_information)
                hydrate.assert_not_called()

    def test_difference_applies_scope_before_mask_guard(self):
        self.add(11, "10000001", 1)
        self.add(21, "10000002", 2)
        self.add(22, "9064xxxx", 2, year=2)
        result, hydrate = self.compare("set_difference")
        self.assert_codes(result, ["10000001"])
        hydrate.assert_called_once_with(self.db, (11, 21))

    def test_unknown_non_course_code_cannot_prove_membership_or_absence(self):
        self.add(11, "10000001", 1)
        self.add(21, "unresolved", 2)
        for operation in ("overlap", "set_difference"):
            with self.subTest(operation=operation):
                result, hydrate = self.compare(operation)
                self.assertEqual(result.status, "missing_data")
                hydrate.assert_not_called()

    def test_bound_applies_after_temporal_filtering_and_deduplication(self):
        for index in range(201):
            self.add(index + 1, f"{10000000 + index:08d}", 1, year=2)
        self.add(300, "10000300", 1)
        for _ in range(201):
            self.add(300, "10000300", 3)
        identities, error = _side_course_set(self.db, self.left)
        self.assertIsNone(error)
        self.assertEqual(identities, (("10000300", 300),))
        year_two = replace(self.left, scope=replace(self.left.scope, years=(2,)))
        identities, error = _side_course_set(self.db, year_two)
        self.assertIsNone(identities)
        self.assertEqual(error, "course set exceeds bounded comparison size")

    def test_exactly_two_hundred_concrete_identities_remain_supported(self):
        for index in range(200):
            self.add(index + 1, f"{10000000 + index:08d}", 1)
        identities, error = _side_course_set(self.db, self.left)
        self.assertIsNone(error)
        self.assertEqual(len(identities), 200)

    def test_unknown_temporal_membership_cannot_prove_absence(self):
        self.add(11, "10000001", 1)
        self.add(21, "10000001", 2, year=None, semester=None, flexible="unknown")
        result, hydrate = self.compare("set_difference")
        self.assertEqual(result.status, "missing_data")
        self.assertIn("course set contains unverified temporal membership", result.missing_information)
        hydrate.assert_not_called()


class CanonicalYearOneOverlapTests(unittest.TestCase):
    def test_exact_student_question_reaches_verified_year_one_overlap(self):
        question = "DSBA 2565 ปี 1 กับ AIT 2566 ปี 1 มีวิชาไหนเรียนร่วมกันบ้าง"
        payload = dict(
            task="compare", subject="course", relation=None,
            target=dict(kind="none", raw_text=None, normalized_hint=None, ordinal=None),
            scope=dict(program=None, catalog=None, plan=None, plan_hint=None, year=None, semester=None),
            filters=[], aggregation=None, ranking=None,
            comparison=dict(
                left=dict(program="DSBA", catalog="2565", year=1),
                right=dict(program="AIT", catalog="2566", year=1),
                measure="course_count", operation="overlap"),
            requested_fields=["code", "name"], clarification=None,
            policy_topic=None, observed_value=None,
        )
        captured = []

        def execute(db_path, resolved):
            verified = execute_comparison(db_path, resolved)
            captured.append((resolved, verified))
            return verified

        with patch("rag.semantic.pipeline.execute_comparison", side_effect=execute):
            response = semantic_answer(DB, question, interpret_callable=lambda _: json.dumps(payload))
        self.assertEqual(response.result.status, "answer", response.trace.failure_reason)
        resolved, verified = captured[0]
        self.assertEqual(
            [(side.scope.program, side.scope.catalog_key, side.scope.years)
             for side in resolved.comparison_sides],
            [("DSBA", "dsba-2565", (1,)), ("AIT", "ait-2566", (1,))],
        )
        self.assertEqual(re.findall(r"\b[0-9]{8}\b", "\n".join(verified.summary_facts)),
                         ["06066001", "06066303"])
        self.assertEqual(re.findall(r"\b[0-9]{8}\b", response.result.final_answer),
                         ["06066001", "06066303"])
        self.assertTrue(verified.provenance)


if __name__ == "__main__":
    unittest.main()
