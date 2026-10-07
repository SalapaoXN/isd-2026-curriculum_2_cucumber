"""Focused tests for deterministic SQL-row provenance hydration (HSQL-1)."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from rag.structured.provenance import (
    SqlRowProvenanceResult,
    hydrate_sql_row_provenance,
)


def _make_db() -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = handle.name
    handle.close()
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE provenance (
                provenance_id INTEGER PRIMARY KEY,
                source_document_key TEXT NOT NULL,
                program TEXT,
                source_filename TEXT,
                source_page INTEGER,
                document_page INTEGER,
                document_category TEXT NOT NULL DEFAULT 'unknown',
                source_uri TEXT,
                source_locator TEXT,
                excerpt TEXT
            );
            CREATE TABLE course_provenance (
                course_id INTEGER NOT NULL,
                provenance_id INTEGER NOT NULL,
                PRIMARY KEY (course_id, provenance_id)
            );
            CREATE TABLE plan_placement_provenance (
                placement_id INTEGER NOT NULL,
                provenance_id INTEGER NOT NULL,
                PRIMARY KEY (placement_id, provenance_id)
            );
            CREATE TABLE prerequisite_provenance (
                prerequisite_id INTEGER NOT NULL,
                provenance_id INTEGER NOT NULL,
                PRIMARY KEY (prerequisite_id, provenance_id)
            );
            CREATE TABLE program_requirement_provenance (
                requirement_id INTEGER NOT NULL,
                provenance_id INTEGER NOT NULL,
                PRIMARY KEY (requirement_id, provenance_id)
            );
            CREATE TABLE policy_fact_provenance (
                fact_id INTEGER NOT NULL,
                provenance_id INTEGER NOT NULL,
                PRIMARY KEY (fact_id, provenance_id)
            );
            CREATE TABLE alternative_group_provenance (
                alternative_group_id INTEGER NOT NULL,
                provenance_id INTEGER NOT NULL,
                PRIMARY KEY (alternative_group_id, provenance_id)
            );
            """
        )
        connection.executemany(
            "INSERT INTO provenance (provenance_id, source_document_key, program,"
            " source_filename, source_page) VALUES (?, ?, ?, ?, ?)",
            [
                (1, "doc-a", "IT", "it_page_001.png", 1),
                (2, "doc-a", "IT", "it_page_002.png", 2),
                (3, "doc-b", "DSBA", "dsba_page_010.png", 10),
            ],
        )
        connection.executemany(
            "INSERT INTO course_provenance (course_id, provenance_id)"
            " VALUES (?, ?)",
            [(101, 1), (101, 2), (102, 3)],
        )
        connection.execute(
            "INSERT INTO plan_placement_provenance (placement_id, provenance_id)"
            " VALUES (201, 2)"
        )
        connection.execute(
            "INSERT INTO prerequisite_provenance (prerequisite_id, provenance_id)"
            " VALUES (301, 3)"
        )
        connection.execute(
            "INSERT INTO program_requirement_provenance"
            " (requirement_id, provenance_id) VALUES (401, 1)"
        )
        connection.execute(
            "INSERT INTO policy_fact_provenance (fact_id, provenance_id)"
            " VALUES (501, 3)"
        )
        connection.execute(
            "INSERT INTO alternative_group_provenance"
            " (alternative_group_id, provenance_id) VALUES (601, 2)"
        )
        connection.commit()
    finally:
        connection.close()
    return path


class SqlRowProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = _make_db()

    @classmethod
    def tearDownClass(cls):
        Path(cls.db_path).unlink(missing_ok=True)

    def test_course_id_resolves_canonical_provenance(self):
        result = hydrate_sql_row_provenance(
            self.db_path, [{"course_id": 101, "name_th": "..."}]
        )
        self.assertIsInstance(result, SqlRowProvenanceResult)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.covered_rows, 1)
        self.assertEqual(result.total_rows, 1)
        self.assertEqual(
            [reference["provenance_id"] for reference in result.provenance],
            [1, 2],
        )
        self.assertEqual(result.provenance[0]["source_filename"], "it_page_001.png")
        self.assertEqual(result.provenance[0]["source_page"], 1)

    def test_placement_id_resolves_placement_provenance(self):
        result = hydrate_sql_row_provenance(self.db_path, [{"placement_id": 201}])
        self.assertEqual(result.status, "complete")
        self.assertEqual(
            [reference["provenance_id"] for reference in result.provenance], [2]
        )

    def test_prerequisite_id_resolves_prerequisite_provenance(self):
        result = hydrate_sql_row_provenance(self.db_path, [{"prerequisite_id": 301}])
        self.assertEqual(result.status, "complete")
        self.assertEqual(
            [reference["provenance_id"] for reference in result.provenance], [3]
        )

    def test_requirement_id_resolves_program_requirement_provenance(self):
        result = hydrate_sql_row_provenance(self.db_path, [{"requirement_id": 401}])
        self.assertEqual(result.status, "complete")
        self.assertEqual(
            [reference["provenance_id"] for reference in result.provenance], [1]
        )

    def test_fact_id_resolves_policy_fact_provenance(self):
        result = hydrate_sql_row_provenance(self.db_path, [{"fact_id": 501}])
        self.assertEqual(result.status, "complete")
        self.assertEqual(
            [reference["provenance_id"] for reference in result.provenance], [3]
        )

    def test_alternative_group_id_resolves_group_provenance(self):
        result = hydrate_sql_row_provenance(
            self.db_path, [{"alternative_group_id": 601}]
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(
            [reference["provenance_id"] for reference in result.provenance], [2]
        )

    def test_all_rows_covered_reports_complete(self):
        result = hydrate_sql_row_provenance(
            self.db_path,
            [{"course_id": 101}, {"placement_id": 201}, {"fact_id": 501}],
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.covered_rows, 3)
        self.assertEqual(result.total_rows, 3)

    def test_mixed_coverage_is_never_complete(self):
        result = hydrate_sql_row_provenance(
            self.db_path, [{"course_id": 101}, {"name_th": "no id here"}]
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.covered_rows, 1)
        self.assertEqual(result.total_rows, 2)

    def test_dangling_id_is_insufficient_evidence(self):
        result = hydrate_sql_row_provenance(self.db_path, [{"course_id": 999}])
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.provenance, ())
        self.assertEqual(result.covered_rows, 0)

    def test_boolean_id_is_rejected(self):
        result = hydrate_sql_row_provenance(self.db_path, [{"course_id": True}])
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.provenance, ())

    def test_non_positive_ids_are_rejected(self):
        for bad in (0, -7):
            with self.subTest(bad=bad):
                result = hydrate_sql_row_provenance(
                    self.db_path, [{"course_id": bad}]
                )
                self.assertEqual(result.status, "insufficient_evidence")

    def test_duplicate_references_deduplicated_in_first_seen_order(self):
        result = hydrate_sql_row_provenance(
            self.db_path, [{"placement_id": 201}, {"course_id": 101}]
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(
            [reference["provenance_id"] for reference in result.provenance],
            [2, 1],
        )

    def test_multiple_supporting_ids_in_one_row_union_in_order(self):
        result = hydrate_sql_row_provenance(
            self.db_path, [{"course_id": 102, "placement_id": 201}]
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(
            [reference["provenance_id"] for reference in result.provenance],
            [3, 2],
        )

    def test_empty_row_set_is_safe_empty_status(self):
        result = hydrate_sql_row_provenance(self.db_path, [])
        self.assertEqual(result.status, "valid_empty")
        self.assertEqual(result.provenance, ())
        self.assertEqual(result.covered_rows, 0)
        self.assertEqual(result.total_rows, 0)

    def test_fake_row_provenance_field_is_ignored(self):
        result = hydrate_sql_row_provenance(
            self.db_path,
            [{"course_id": 101, "provenance": [{"source_page": 999}]}],
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(
            [reference["provenance_id"] for reference in result.provenance],
            [1, 2],
        )
        self.assertNotIn(
            999,
            [
                reference.get("source_page")
                for reference in result.provenance
            ],
        )

    def test_unrecognized_id_columns_do_not_cover(self):
        result = hydrate_sql_row_provenance(
            self.db_path, [{"catalog_id": 1, "plan_id": 2}]
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.covered_rows, 0)


if __name__ == "__main__":
    unittest.main()
