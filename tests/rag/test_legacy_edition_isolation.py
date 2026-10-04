"""Legacy curriculum edition isolation (IT/BIT 2560, GENED 2557 vs current editions).

Proves catalog-scoped separation for the integrated legacy OCR datasets:
same logical program across editions shares nothing except the program
name; plans stay plans; both GENED editions coexist distinctly; unknown
editions (including the refuted it-2559) fail closed.
"""
import re
import sqlite3
import tempfile
import unittest
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

from rag.policy import answer_policy_question
from rag.policy.repository import fetch_program_requirement
from rag.qa import ask
from rag.resolution import QueryContext
from rag.structured.loader import load_jsons_to_sqlite


DB_PATH = Path(__file__).resolve().parents[2] / "cucumber_outputs" / "runtime" / "curriculum.db"
FINAL_DIR = Path(__file__).resolve().parents[2] / "data" / "output" / "final"

LEGACY_IT = "it-2560"
CURRENT_IT = "it-2565"
LEGACY_BIT = "bit-2560"
CURRENT_BIT = "bit-2565"

LEGACY_FILES = [
    FINAL_DIR / "merged_it2560_no_coop_edition-it-2560_full_corrected.json",
    FINAL_DIR / "merged_it2560_coop_edition-it-2560_full_corrected.json",
    FINAL_DIR / "merged_bit2560_no_coop_edition-bit-2560_full_corrected.json",
    FINAL_DIR / "merged_bit2560_coop_edition-bit-2560_full_corrected.json",
    FINAL_DIR / "merged_gened2557_gened_edition-gened-2557_full_corrected.json",
]

_CODE_RE = re.compile(r"'course_code': '([0-9A-Za-zXx]+)'")


def _failing_model(prompt):
    raise AssertionError("deterministic isolation case must not call a model")


def _query_db(statement, parameters=()):
    with closing(sqlite3.connect(f"file:{DB_PATH.resolve()}?mode=ro", uri=True)) as connection:
        return connection.execute(statement, parameters).fetchall()


def _claim_codes(result):
    codes = set()
    claims = result.claims if hasattr(result, "claims") else []
    for claim in claims:
        value = claim.value
        entries = value if isinstance(value, (list, tuple)) else [value]
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("course_code"):
                codes.add(entry["course_code"])
            elif isinstance(entry, str):
                codes.update(_CODE_RE.findall(entry))
    return codes


class LegacyCatalogIdentityTests(unittest.TestCase):
    def test_legacy_catalogs_exist_with_verified_academic_years(self):
        rows = dict(
            (key, year)
            for key, year in _query_db(
                "SELECT catalog_key, academic_year FROM catalogs"
            )
        )
        # Edition years verified from source: IT cover
        # '(หลักสูตรปรับปรุง พ.ศ. 2560)', BIT cover '(หลักสูตรใหม่ พ.ศ. 2560)',
        # GENED 2557 book colophon/approvals, GENED 2564 revision structure.
        # The 2559 dates on the IT/BIT covers are council approval dates,
        # not the curriculum edition: it-2559 must NOT exist.
        self.assertEqual(rows[LEGACY_IT], "2560")
        self.assertEqual(rows[LEGACY_BIT], "2560")
        self.assertEqual(rows[CURRENT_IT], "2565")
        self.assertEqual(rows[CURRENT_BIT], "2565")
        self.assertEqual(rows["gened-2557"], "2557")
        self.assertEqual(rows["gened-2564"], "2564")
        self.assertNotIn("it-2559", rows)
        self.assertNotIn("it-2560-coop", rows)

    def test_coop_and_no_coop_remain_plans_not_editions(self):
        rows = _query_db(
            """SELECT c.catalog_key, cp.plan_key FROM curriculum_plans cp
               JOIN catalogs c ON c.catalog_id = cp.catalog_id
               WHERE c.catalog_key IN ('it-2560', 'bit-2560')
               ORDER BY 1, 2"""
        )
        self.assertEqual(
            rows,
            [
                ("bit-2560", "coop"),
                ("bit-2560", "no_coop"),
                ("it-2560", "coop"),
                ("it-2560", "no_coop"),
            ],
        )
        catalog_keys = [row[0] for row in _query_db("SELECT catalog_key FROM catalogs")]
        self.assertFalse(any("coop" in key for key in catalog_keys))

    def test_gened_editions_coexist_without_collision(self):
        rows = _query_db(
            "SELECT catalog_key, COUNT(*) FROM catalogs WHERE catalog_key LIKE 'gened%' "
            "GROUP BY catalog_key"
        )
        self.assertEqual(rows, [("gened-2557", 1), ("gened-2564", 1)])
        # Legacy 2557 book: 90-family codes from gened2557_page_* sources.
        legacy = _query_db(
            """SELECT COUNT(*) FROM courses JOIN catalogs USING (catalog_id)
               WHERE catalog_key = 'gened-2557' AND course_code LIKE '90%'"""
        )
        total_legacy = _query_db(
            """SELECT COUNT(*) FROM courses JOIN catalogs USING (catalog_id)
               WHERE catalog_key = 'gened-2557'"""
        )
        self.assertEqual(total_legacy, [(123,)])
        self.assertEqual(legacy, total_legacy)
        legacy_sources = _query_db(
            """SELECT COUNT(DISTINCT p.source_filename) FROM provenance p
               JOIN course_provenance cp ON cp.provenance_id = p.provenance_id
               JOIN courses co ON co.course_id = cp.course_id
               JOIN catalogs c ON c.catalog_id = co.catalog_id
               WHERE c.catalog_key = 'gened-2557'"""
        )
        self.assertGreater(legacy_sources[0][0], 0)
        # Current 2564 book: 9064-code courses from gened_page_* sources.
        current = _query_db(
            """SELECT COUNT(*) FROM courses JOIN catalogs USING (catalog_id)
               WHERE catalog_key = 'gened-2564' AND course_code LIKE '9064%'"""
        )
        total_current = _query_db(
            """SELECT COUNT(*) FROM courses JOIN catalogs USING (catalog_id)
               WHERE catalog_key = 'gened-2564'"""
        )
        self.assertEqual(total_current, [(266,)])
        self.assertEqual(current, total_current)
        overlap = _query_db(
            """SELECT COUNT(*) FROM courses co1 JOIN courses co2
               ON co1.course_code = co2.course_code
               JOIN catalogs c1 ON c1.catalog_id = co1.catalog_id
               JOIN catalogs c2 ON c2.catalog_id = co2.catalog_id
               WHERE c1.catalog_key = 'gened-2557' AND c2.catalog_key = 'gened-2564'"""
        )
        self.assertEqual(overlap, [(0,)])


class NoCrossEditionLeakageTests(unittest.TestCase):
    def test_it_course_codes_do_not_leak_across_editions(self):
        legacy_only = _query_db(
            """SELECT DISTINCT course_code FROM courses JOIN catalogs USING (catalog_id)
               WHERE catalog_key = 'it-2560' AND course_code LIKE '060163%'"""
        )
        self.assertTrue(legacy_only)
        leaked = _query_db(
            """SELECT COUNT(*) FROM courses JOIN catalogs USING (catalog_id)
               WHERE catalog_key = 'it-2565' AND course_code LIKE '060163%'"""
        )
        self.assertEqual(leaked, [(0,)])
        current_leaked = _query_db(
            """SELECT COUNT(*) FROM courses JOIN catalogs USING (catalog_id)
               WHERE catalog_key = 'it-2560' AND course_code LIKE '060164%'"""
        )
        self.assertEqual(current_leaked, [(0,)])

    def test_bit_course_codes_do_not_leak_across_editions(self):
        legacy_only = _query_db(
            """SELECT DISTINCT course_code FROM courses JOIN catalogs USING (catalog_id)
               WHERE catalog_key = 'bit-2560' AND course_code LIKE '060360%'"""
        )
        self.assertTrue(legacy_only)
        self.assertEqual(
            _query_db(
                """SELECT COUNT(*) FROM courses JOIN catalogs USING (catalog_id)
                   WHERE catalog_key = 'bit-2565' AND course_code LIKE '060360%'"""
            ),
            [(0,)],
        )
        self.assertEqual(
            _query_db(
                """SELECT COUNT(*) FROM courses JOIN catalogs USING (catalog_id)
                   WHERE catalog_key = 'bit-2560' AND course_code LIKE '060361%'"""
            ),
            [(0,)],
        )

    def test_shared_gened_code_stays_in_legacy_era_editions(self):
        # 90101007 is a legacy-era plan-table course shared by the 2560
        # editions and the 2557 GENED book; it must not resolve in the
        # current IT edition.
        rows = _query_db(
            """SELECT c.catalog_key FROM courses co JOIN catalogs c USING (catalog_id)
               WHERE co.course_code = '90101007' ORDER BY 1"""
        )
        self.assertEqual(rows, [("dsba-2560",), ("gened-2557",), ("it-2560",)])
        self.assertNotIn(
            ("it-2565",),
            _query_db(
                """SELECT c.catalog_key FROM courses co JOIN catalogs c USING (catalog_id)
                   WHERE co.course_code = '90101007' AND c.catalog_key = 'it-2565'"""
            ),
        )

    def test_legacy_coop_course_absent_from_current_edition(self):
        rows = _query_db(
            """SELECT c.catalog_key FROM courses co JOIN catalogs c USING (catalog_id)
               WHERE co.course_code = '06036046' ORDER BY 1"""
        )
        self.assertEqual(rows, [("bit-2560",)])

    def test_plan_placements_stay_inside_their_catalog(self):
        rows = _query_db(
            """SELECT COUNT(*) FROM plan_placements pp
               JOIN curriculum_plans cp ON cp.plan_id = pp.plan_id
               JOIN courses co ON co.course_id = pp.course_id
               JOIN catalogs plan_catalog ON plan_catalog.catalog_id = cp.catalog_id
               JOIN catalogs course_catalog ON course_catalog.catalog_id = co.catalog_id
               WHERE plan_catalog.catalog_key IN ('it-2560', 'bit-2560')
                 AND plan_catalog.catalog_id != course_catalog.catalog_id"""
        )
        self.assertEqual(rows, [(0,)])

    def test_prerequisites_stay_edition_scoped(self):
        rows = _query_db(
            """SELECT COUNT(*) FROM prerequisites p
               JOIN courses dependent ON dependent.course_id = p.course_id
               JOIN courses required ON required.course_id = p.prerequisite_course_id
               JOIN catalogs c1 ON c1.catalog_id = dependent.catalog_id
               JOIN catalogs c2 ON c2.catalog_id = required.catalog_id
               WHERE c1.catalog_key IN ('it-2560', 'bit-2560')
                 AND c1.catalog_id != c2.catalog_id"""
        )
        self.assertEqual(rows, [(0,)])
        legacy_prereqs = _query_db(
            """SELECT COUNT(*) FROM prerequisites p
               JOIN courses co ON co.course_id = p.course_id
               JOIN catalogs c ON c.catalog_id = co.catalog_id
               WHERE c.catalog_key IN ('it-2560', 'bit-2560')"""
        )
        self.assertGreater(legacy_prereqs[0][0], 0)

    def test_legacy_provenance_references_legacy_source_pages(self):
        rows = _query_db(
            """SELECT DISTINCT p.source_filename FROM provenance p
               JOIN course_provenance cp ON cp.provenance_id = p.provenance_id
               JOIN courses co ON co.course_id = cp.course_id
               JOIN catalogs c ON c.catalog_id = co.catalog_id
               WHERE c.catalog_key IN ('it-2560', 'bit-2560')"""
        )
        filenames = [name for (name,) in rows]
        self.assertTrue(filenames)
        for name in filenames:
            self.assertTrue(
                name.startswith("it2560_page_") or name.startswith("bit2560_page_"),
                name,
            )


class LegacyRequirementTests(unittest.TestCase):
    def test_legacy_totals_are_edition_scoped(self):
        self.assertEqual(
            fetch_program_requirement(DB_PATH, "IT", catalog_key="it-2560")["value"],
            130,
        )
        self.assertEqual(
            fetch_program_requirement(DB_PATH, "BIT", catalog_key="bit-2560")["value"],
            126,
        )

    def test_current_totals_are_unchanged(self):
        self.assertEqual(
            fetch_program_requirement(DB_PATH, "IT", catalog_key="it-2565")["value"],
            129,
        )
        self.assertEqual(
            fetch_program_requirement(DB_PATH, "BIT", catalog_key="bit-2565")["value"],
            126,
        )
        self.assertEqual(
            fetch_program_requirement(DB_PATH, "DSBA", catalog_key="dsba-2560")["value"],
            126,
        )
        self.assertEqual(
            fetch_program_requirement(DB_PATH, "DSBA", catalog_key="dsba-2565")["value"],
            132,
        )

    def test_gened_has_no_degree_program_total(self):
        self.assertIsNone(fetch_program_requirement(DB_PATH, "GENED"))
        self.assertIsNone(
            fetch_program_requirement(DB_PATH, "GENED", catalog_key="gened-2557")
        )
        self.assertIsNone(
            fetch_program_requirement(DB_PATH, "GENED", catalog_key="gened-2564")
        )


class LegacyQaIsolationTests(unittest.TestCase):
    def test_ambiguous_program_question_asks_for_catalog(self):
        response = ask(
            DB_PATH,
            "IT มีวิชาอะไรบ้าง",
            intent_model_callable=_failing_model,
            structured_model_callable=_failing_model,
            answer_model_callable=None,
        )
        result = response["result"]
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("status"), "clarify_catalog")
        self.assertEqual(
            sorted(result.get("catalog_keys", [])), ["it-2560", "it-2565"]
        )

    def test_unknown_edition_fails_closed(self):
        response = ask(
            DB_PATH,
            "IT มีวิชาอะไรบ้าง",
            conversation_context=QueryContext(program="IT", catalog_key="it-2559"),
            intent_model_callable=_failing_model,
            structured_model_callable=_failing_model,
            answer_model_callable=None,
        )
        result = response["result"]
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("status"), "clarify_catalog")

    def test_scoped_year_list_returns_only_edition_codes(self):
        legacy = ask(
            DB_PATH,
            "IT ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
            conversation_context=QueryContext(program="IT", catalog_key="it-2560"),
            intent_model_callable=_failing_model,
            structured_model_callable=_failing_model,
            answer_model_callable=None,
        )
        legacy_codes = _claim_codes(legacy["result"])
        self.assertTrue(legacy_codes)
        self.assertTrue(all(code.startswith("060163") or code[:3] in ("901", "902", "903", "904") for code in legacy_codes))
        current = ask(
            DB_PATH,
            "IT ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
            conversation_context=QueryContext(program="IT", catalog_key="it-2565"),
            intent_model_callable=_failing_model,
            structured_model_callable=_failing_model,
            answer_model_callable=None,
        )
        current_codes = _claim_codes(current["result"])
        self.assertTrue(current_codes)
        self.assertEqual(legacy_codes & current_codes, set())

    def test_legacy_course_query_is_grounded_in_legacy_source(self):
        response = ask(
            DB_PATH,
            "06016301 คือวิชาอะไร",
            intent_model_callable=_failing_model,
            structured_model_callable=_failing_model,
            answer_model_callable=None,
        )
        result = response["result"]
        self.assertEqual(result.status, "answer")
        filenames = [
            reference.get("source_filename", "")
            for reference in (result.provenance or [])
        ]
        self.assertTrue(filenames)
        self.assertTrue(all(name.startswith("it2560_page_") for name in filenames))

    def test_legacy_policy_totals_are_catalog_scoped(self):
        legacy = answer_policy_question(DB_PATH, "BIT ต้องเรียนกี่หน่วยกิต", catalog_key="bit-2560")
        self.assertEqual((legacy.status, legacy.value), ("complete", 126))
        current = answer_policy_question(DB_PATH, "BIT ต้องเรียนกี่หน่วยกิต", catalog_key="bit-2565")
        self.assertEqual((current.status, current.value), ("complete", 126))
        ambiguous = answer_policy_question(DB_PATH, "BIT ต้องเรียนกี่หน่วยกิต")
        self.assertEqual(ambiguous.status, "insufficient_evidence")

    def test_gened_old_new_comparison_resolves_both_editions(self):
        from backend.hard_qa import answer_hard_question

        def _never(prompt):
            raise AssertionError("edition pair must not call interpreter")

        result = answer_hard_question(
            DB_PATH,
            "วิชาที่มีในหลักสูตรเก่า GENED ไม่มีในหลักสูตรใหม่มีอะไรบ้าง",
            {"program": "GENED"},
            _never,
        )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["hard_task_type"], "old_new_comparison")
        self.assertIn("gened-2557", result["answer"])
        self.assertIn("gened-2564", result["answer"])
        self.assertTrue(
            any(
                str(reference.get("source_filename", "")).startswith("gened2557_page_")
                for reference in result["provenance"]
            )
        )


class LegacyRegressionGuardTests(unittest.TestCase):
    def test_dsba_and_ait_catalogs_are_unchanged(self):
        rows = dict(
            (key, count)
            for key, count in _query_db(
                """SELECT c.catalog_key, COUNT(*) FROM courses co
                   JOIN catalogs c USING (catalog_id)
                   WHERE c.catalog_key IN ('dsba-2560', 'dsba-2565', 'ait-2566')
                   GROUP BY c.catalog_key"""
            )
        )
        self.assertEqual(rows, {"dsba-2560": 80, "dsba-2565": 90, "ait-2566": 55})

    def test_legacy_artifacts_load_standalone_with_two_plans_each(self):
        for path in LEGACY_FILES:
            self.assertTrue(path.is_file(), path)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "legacy.db"
            catalog_ids = load_jsons_to_sqlite(LEGACY_FILES, database)
            self.assertEqual(len(catalog_ids), 5)
            with closing(sqlite3.connect(database)) as connection:
                catalogs = connection.execute(
                    "SELECT catalog_key, academic_year FROM catalogs ORDER BY catalog_key"
                ).fetchall()
                plans = connection.execute(
                    """SELECT c.catalog_key, cp.plan_key FROM curriculum_plans cp
                       JOIN catalogs c ON c.catalog_id = cp.catalog_id
                       ORDER BY 1, 2"""
                ).fetchall()
        self.assertEqual(
            catalogs, [("bit-2560", "2560"), ("gened-2557", "2557"), ("it-2560", "2560")]
        )
        self.assertEqual(
            plans,
            [
                ("bit-2560", "coop"),
                ("bit-2560", "no_coop"),
                ("gened-2557", "gened"),
                ("it-2560", "coop"),
                ("it-2560", "no_coop"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
