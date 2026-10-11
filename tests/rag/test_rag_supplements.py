import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

from rag.structured.supplements import load_rule_supplements


class RuleSupplementTests(unittest.TestCase):
    def test_default_index_build_refreshes_sqlite_rule_supplements(self):
        builder = import_module("rag.build_index")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "it2565_final.json"
            database = root / "curriculum.db"
            source.touch()
            with patch.object(builder, "_PROJECT_ROOT", root), \
                 patch.object(builder, "_default_sources", return_value=[source]), \
                 patch.object(builder, "audit_shared_course_conflicts", return_value=[]), \
                 patch.object(builder, "ensure_index", return_value=database), \
                 patch.object(builder, "load_rule_supplements") as refresh:
                result = builder.build_index()

        self.assertEqual(result, database)
        refresh.assert_called_once_with(
            database,
            root / "data" / "output" / "final" / "institution_policy.json",
            root / "data" / "output" / "final" / "program_requirements.json",
        )

    def test_loads_academic_rules_policy_facts_and_program_requirements_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "curriculum.db"
            policy_path = root / "institution_policy.json"
            requirements_path = root / "program_requirements.json"
            with closing(sqlite3.connect(database)) as db:
                db.executescript("""
                    CREATE TABLE provenance(
                        provenance_id INTEGER PRIMARY KEY, source_document_key TEXT NOT NULL,
                        program TEXT, source_filename TEXT, source_page INTEGER, document_page INTEGER,
                        document_category TEXT NOT NULL, source_uri TEXT, source_locator TEXT, excerpt TEXT);
                    CREATE TABLE regulation_rules(
                        rule_id TEXT PRIMARY KEY, section_number TEXT NOT NULL, parent_rule_id TEXT,
                        category TEXT NOT NULL, rule_text TEXT NOT NULL, references_json TEXT NOT NULL);
                    CREATE TABLE policy_facts(
                        fact_id INTEGER PRIMARY KEY, category TEXT NOT NULL, fact_key TEXT,
                        operator TEXT NOT NULL, value NUMERIC NOT NULL, unit TEXT, condition TEXT,
                        context TEXT, source_rule_id TEXT NOT NULL, verification_status TEXT);
                    CREATE TABLE policy_fact_provenance(
                        fact_id INTEGER NOT NULL, provenance_id INTEGER NOT NULL,
                        PRIMARY KEY(fact_id, provenance_id));
                    CREATE TABLE catalogs(catalog_id INTEGER PRIMARY KEY, catalog_key TEXT);
                    CREATE TABLE program_requirements(
                        requirement_id INTEGER PRIMARY KEY, catalog_id INTEGER NOT NULL,
                        program_code TEXT NOT NULL, requirement_type TEXT NOT NULL,
                        operator TEXT NOT NULL, value NUMERIC NOT NULL, unit TEXT NOT NULL);
                    CREATE TABLE program_requirement_provenance(
                        requirement_id INTEGER NOT NULL, provenance_id INTEGER NOT NULL,
                        PRIMARY KEY(requirement_id, provenance_id));
                    INSERT INTO catalogs VALUES (1, 'it-2565');
                """)
                db.commit()

            provenance = [{
                "program": "RULE", "source_filename": "rules2564_page_007.png",
                "source_page": 7, "document_category": "rule", "source_verified": True,
            }]
            policy_path.write_text(json.dumps({"categories": [{
                "category": "Probation",
                "values": [{"label": "Cumulative GPA probation threshold", "value": "2.00",
                            "condition": "below", "source_rule_id": "rule:22"}],
                "evidence": {"source_provenance": provenance,
                             "supporting_rule_text": [{"rule_id": "rule:22", "rule_text": "GPA below 2.00 results in probation.",
                                                       "snippets": []}]},
            }]}), encoding="utf-8")
            requirements_path.write_text(json.dumps([{
                "program": "IT", "catalog_key": "it-2565", "requirement_type": "total_program_credits",
                "operator": "=", "value": 129, "unit": "credits",
                "source_provenance": [{"program": "IT", "source_filename": "it2565_page_006.png",
                                       "source_page": 6, "document_category": "program_requirement"}],
            }]), encoding="utf-8")

            for _ in range(2):
                counts = load_rule_supplements(database, policy_path, requirements_path)
                self.assertEqual(counts, {"regulation_rules": 1, "policy_facts": 1, "program_requirements": 1})

            with closing(sqlite3.connect(database)) as db:
                rule = db.execute("SELECT rule_text, references_json FROM regulation_rules WHERE rule_id='rule:22'").fetchone()
                fact = db.execute("SELECT operator, value, source_rule_id FROM policy_facts").fetchone()
                requirement = db.execute("SELECT program_code, value FROM program_requirements").fetchone()
                provenance_count = db.execute("SELECT count(*) FROM provenance").fetchone()[0]
            self.assertEqual(rule[0], "GPA below 2.00 results in probation.")
            self.assertEqual(json.loads(rule[1])["source_provenance"][0]["source_page"], 7)
            self.assertEqual(fact, ("<", 2, "rule:22"))
            self.assertEqual(requirement, ("IT", 129))
            self.assertEqual(provenance_count, 2)


if __name__ == "__main__":
    unittest.main()
