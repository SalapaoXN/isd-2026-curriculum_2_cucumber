"""Direct canonical ownership controls; no providers or evidence execution."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rag.semantic.resolver import (
    canonical_catalog_key,
    resolve_comparison_operand,
    valid_plan,
)


class ScopeOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "scope.db"
        with closing(sqlite3.connect(self.db)) as connection:
            connection.executescript("""
                CREATE TABLE catalogs (
                    catalog_id INTEGER PRIMARY KEY, catalog_key TEXT,
                    academic_year TEXT);
                CREATE TABLE programs (
                    program_id INTEGER PRIMARY KEY, catalog_id INTEGER,
                    program_code TEXT);
                CREATE TABLE curriculum_plans (
                    catalog_id INTEGER, program_id INTEGER, plan_key TEXT);
                INSERT INTO catalogs VALUES
                    (1, 'it-edition', '2565'),
                    (2, 'dsba-edition', '2565'),
                    (3, 'it-other', '2565');
                INSERT INTO programs VALUES
                    (1, 1, 'IT'), (2, 2, 'DSBA'), (3, 3, 'IT');
                INSERT INTO curriculum_plans VALUES
                    (1, 1, 'coop'), (2, 2, 'foreign-only');
            """)

    def test_matching_catalog(self):
        self.assertEqual(canonical_catalog_key(self.db, "it-edition", "IT"),
                         "it-edition")

    def test_matching_plan(self):
        self.assertEqual(valid_plan(self.db, "coop", "IT", "it-edition"), "coop")

    def test_unique_academic_year(self):
        self.assertEqual(canonical_catalog_key(self.db, "2565", "DSBA"),
                         "dsba-edition")

    def test_dsba_rejects_it_catalog(self):
        self.assertIsNone(canonical_catalog_key(self.db, "it-edition", "DSBA"))
        self.assertIsNone(valid_plan(self.db, "coop", "DSBA", "it-edition"))
        side = resolve_comparison_operand(
            self.db, (("program", "DSBA"), ("catalog", "it-edition")), None, None)
        self.assertTrue(side.unresolved)

    def test_it_rejects_dsba_catalog(self):
        self.assertIsNone(canonical_catalog_key(self.db, "dsba-edition", "IT"))
        side = resolve_comparison_operand(
            self.db, (("program", "IT"), ("catalog", "dsba-edition")), None, None)
        self.assertTrue(side.unresolved)

    def test_foreign_plan_rejected(self):
        self.assertIsNone(valid_plan(self.db, "foreign-only", "IT", "it-edition"))
        side = resolve_comparison_operand(
            self.db, (("program", "IT"), ("catalog", "it-edition"),
                      ("plan", "foreign-only")), None, None)
        self.assertTrue(side.unresolved)

    def test_ambiguous_academic_year(self):
        self.assertIsNone(canonical_catalog_key(self.db, "2565", "IT"))
