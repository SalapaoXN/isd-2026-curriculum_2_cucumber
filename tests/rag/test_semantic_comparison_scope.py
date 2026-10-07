"""Six direct operand controls for side-local comparison scope."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from rag.semantic.resolver import resolve_comparison_operand


class ComparisonScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "scope.db"
        with closing(sqlite3.connect(self.db)) as connection:
            connection.executescript("""
                CREATE TABLE catalogs (catalog_id INTEGER, catalog_key TEXT,
                                       academic_year TEXT);
                CREATE TABLE programs (program_id INTEGER, catalog_id INTEGER,
                                       program_code TEXT);
                CREATE TABLE curriculum_plans (program_id INTEGER,
                                              catalog_id INTEGER, plan_key TEXT);
                INSERT INTO catalogs VALUES
                    (1, 'it-2565', '2565'), (2, 'dsba-2565', '2565'),
                    (3, 'dsba-2560', '2560');
                INSERT INTO programs VALUES
                    (1, 1, 'IT'), (2, 2, 'DSBA'), (3, 3, 'DSBA');
                INSERT INTO curriculum_plans VALUES (1, 1, 'coop');
            """)

    def resolve(self, **side):
        return resolve_comparison_operand(
            self.db, tuple(side.items()), "IT", "it-2565", (1,), (1,))

    def test_compatible_catalog_inheritance(self):
        operand = self.resolve(program="IT")
        self.assertFalse(operand.unresolved)
        self.assertEqual(operand.scope.catalog_key, "it-2565")

    def test_explicit_independent_catalogs(self):
        left = self.resolve(program="IT", catalog="it-2565")
        right = self.resolve(program="DSBA", catalog="dsba-2565")
        self.assertFalse(left.unresolved)
        self.assertFalse(right.unresolved)
        self.assertEqual(right.scope.program, "DSBA")
        self.assertEqual(right.scope.catalog_key, "dsba-2565")

    def test_side_local_temporal_override(self):
        operand = self.resolve(program="IT", year=2, semester=2)
        self.assertFalse(operand.unresolved)
        self.assertEqual((operand.scope.years, operand.scope.semesters), ((2,), (2,)))

    def test_foreign_default_requires_catalog_clarification(self):
        operand = self.resolve(program="DSBA")
        self.assertTrue(operand.unresolved)
        self.assertEqual(operand.reason, "ambiguous operand catalog")
        self.assertNotEqual(operand.scope.catalog_key, "it-2565")

    def test_explicit_foreign_catalog_rejected(self):
        operand = self.resolve(program="DSBA", catalog="it-2565")
        self.assertTrue(operand.unresolved)
        self.assertEqual(operand.reason, "unknown operand catalog")

    def test_no_common_plan_inheritance(self):
        # The resolver accepts no default-plan argument: even the canonical
        # home catalog's coop plan must not be selected for either side.
        home = self.resolve(program="IT")
        foreign = self.resolve(program="DSBA", catalog="dsba-2565")
        self.assertFalse(foreign.unresolved)
        self.assertIsNone(home.scope.plan)
        self.assertIsNone(foreign.scope.plan)
