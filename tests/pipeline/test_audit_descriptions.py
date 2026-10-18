import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from src.pipeline.tools.audit_descriptions import audit, has_strict_failures, normalize_code


class AuditDescriptionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.canonical = self.root / "canonical"
        self.canonical.mkdir()
        self.database = self.root / "curriculum.db"
        with closing(sqlite3.connect(self.database)) as db:
            db.executescript("""
                CREATE TABLE catalogs(catalog_id INTEGER, catalog_key TEXT);
                CREATE TABLE courses(catalog_id INTEGER, course_id INTEGER, course_code TEXT,
                    course_identity_discriminator TEXT, description_th TEXT, description_en TEXT);
                CREATE TABLE course_provenance(course_id INTEGER, provenance_id INTEGER);
                CREATE TABLE provenance(provenance_id INTEGER, document_category TEXT,
                    source_filename TEXT, source_page INTEGER, document_page INTEGER);
                INSERT INTO catalogs VALUES (1, 'it-2565');
            """)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, courses):
        (self.canonical / name).write_text(json.dumps({"catalog": {"catalog_key": "it-2565"}, "courses": courses}), encoding="utf-8")

    def test_reports_mismatch_and_missing_without_claiming_source_absence(self):
        self.write("courses.json", [
            {"code": " C101 ", "desc_th": "canonical", "desc_en": "EN", "source_provenance": [{"document_category": "description", "source_filename": "scan.png", "document_page": 4}]},
            {"code": "C102", "desc_th": "", "desc_en": "", "source_provenance": [{"document_category": "description", "source_filename": "scan.png", "document_page": 5}]},
        ])
        with closing(sqlite3.connect(self.database)) as db:
            db.executemany("INSERT INTO courses VALUES (1, ?, ?, '', ?, ?)", [(1, "C101", "database", "EN"), (2, "C102", "", "")])
            db.execute("INSERT INTO provenance VALUES (1, 'description', 'scan.png', 4, 4)")
            db.execute("INSERT INTO course_provenance VALUES (1, 1)")
            db.commit()
        result = audit(self.canonical, self.database)
        statuses = {row["course_code"]: row["status"] for row in result["courses"]}
        self.assertEqual(statuses, {"c101": "different", "c102": "both_missing"})
        self.assertEqual(result["source_status"], "unknown_without_source_OCR")
        self.assertEqual(result["courses"][0]["canonical"]["provenance"]["categories"], ["description"])
        self.assertTrue(result["courses"][0]["description_source_present"])
        self.assertEqual(result["summary"]["both_missing_with_description_source"], 1)
        c101 = next(row for row in result["courses"] if row["course_code"] == "c101")
        self.assertEqual(c101["database"]["provenance"]["categories"], ["description"])

    def test_per_code_descriptions_avoid_false_conflicts_for_combined_records(self):
        first = "First course description"
        second = "Second course description"
        self.write(
            "courses.json",
            [
                {
                    "code": "C101 or C102",
                    "desc_th": first + "\n" + second,
                    "descriptions_by_code": {
                        "C101": {"desc_th": first, "desc_en": None},
                        "C102": {"desc_th": second, "desc_en": None},
                    },
                }
            ],
        )
        with closing(sqlite3.connect(self.database)) as db:
            db.executemany(
                "INSERT INTO courses VALUES (1, ?, ?, '', ?, ?)",
                [(1, "C101", first, None), (2, "C102", second, None)],
            )
            db.commit()

        result = audit(self.canonical, self.database)

        self.assertEqual(result["summary"]["mismatches"], 0)
        self.assertFalse(has_strict_failures(result))

    def test_same_course_description_is_audited_across_loader_fact_variants(self):
        self.write(
            "courses.json",
            [
                {"code": "C101", "desc_en": "Shared description", "name_en": "OCR TITLE A"},
                {"code": "C101", "desc_en": "Shared description", "name_en": "OCR TITLE B"},
            ],
        )
        with closing(sqlite3.connect(self.database)) as db:
            db.executemany(
                "INSERT INTO courses VALUES (1, ?, 'C101', ?, NULL, 'Shared description')",
                [(1, ""), (2, "variant:second-title")],
            )
            db.commit()

        result = audit(self.canonical, self.database)

        self.assertEqual(result["summary"]["mismatches"], 0)
        self.assertFalse(has_strict_failures(result))

    def test_duplicate_canonical_record_is_ambiguous(self):
        self.write("a.json", [{"code": "A1", "desc_en": "first"}, {"code": "A1", "desc_en": "second"}])
        result = audit(self.canonical, self.database)
        self.assertEqual(result["courses"][0]["status"], "ambiguous")

    def test_normalizes_course_codes(self):
        self.assertEqual(normalize_code(" C10-1 "), "c10-1")

    def test_reports_stale_canonical_fingerprint(self):
        self.write("courses.json", [{"code": "C1", "desc_en": "Description"}])
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("""CREATE TABLE semantic_index_sources(
                source_file_identity TEXT, source_filename TEXT,
                source_json_fingerprint TEXT)""")
            db.execute(
                "INSERT INTO semantic_index_sources VALUES (?, ?, ?)",
                ("repo:data/output/canonical/courses.json", "courses.json", "old-hash"),
            )
            db.commit()

        result = audit(self.canonical, self.database)

        self.assertEqual(len(result["freshness"]["mismatches"]), 1)
        self.assertEqual(
            result["freshness"]["mismatches"][0]["stored_fingerprint"], "old-hash"
        )
        self.assertTrue(has_strict_failures(result))

    def test_strict_mode_allows_expected_missing_without_description_source(self):
        self.write("courses.json", [{"code": "C1", "desc_th": "", "desc_en": ""}])
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("INSERT INTO courses VALUES (1, 1, 'C1', '', '', '')")
            db.commit()

        result = audit(self.canonical, self.database)

        self.assertFalse(has_strict_failures(result))

    def test_freshness_matches_source_filename_exactly_not_suffix(self):
        import hashlib

        files = [
            self.canonical / "bit2560_coop_final.json",
            self.canonical / "it2560_coop_final.json",
        ]
        for path, catalog_key in zip(files, ("bit-2560", "it-2560")):
            path.write_text(
                json.dumps({"catalog": {"catalog_key": catalog_key}, "courses": []}),
                encoding="utf-8",
            )
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("""CREATE TABLE semantic_index_sources(
                source_file_identity TEXT, source_filename TEXT,
                source_json_fingerprint TEXT)""")
            db.executemany(
                "INSERT INTO semantic_index_sources VALUES (?, ?, ?)",
                [
                    (
                        f"repo:data/output/canonical/{path.name}",
                        path.name,
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                    for path in files
                ],
            )
            db.commit()

        result = audit(self.canonical, self.database)

        self.assertEqual(result["freshness"]["mismatches"], [])


if __name__ == "__main__":
    unittest.main()
