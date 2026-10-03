import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rag.structured.loader import load_json_to_sqlite, load_jsons_to_sqlite
from rag.structured.supplemental_loader import load_supplemental_jsons_to_sqlite
from rag.retrieval.index import ensure_index


class RagSupplementalLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.policy = cls.root / "data" / "output" / "final" / "institution_policy.json"
        cls.requirements = (
            cls.root / "data" / "output" / "final" / "program_requirements.json"
        )

    def _curriculum(self, directory: Path) -> Path:
        path = directory / "curriculum.json"
        path.write_text(
            json.dumps(
                {
                    "program": "TEST",
                    "plan": "regular",
                    "catalog": {
                        "catalog_key": "test-catalog",
                        "academic_year": "test",
                    },
                    "courses": [{"code": "C100"}],
                    "source_provenance": [
                        {
                            "source_filename": "curriculum.pdf",
                            "source_page": 1,
                            "document_category": "plan",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def _canonical_curricula(self, directory: Path) -> list[Path]:
        editions = [
            ("DSBA", "dsba-2560", "2560", "coop"),
            ("DSBA", "dsba-2560", "2560", "no_coop"),
            ("DSBA", "dsba-2565", "2565", "coop"),
            ("DSBA", "dsba-2565", "2565", "no_coop"),
            ("IT", "it-2565", "2565", "coop"),
            ("IT", "it-2565", "2565", "no_coop"),
            ("BIT", "bit-2565", "2565", "coop"),
            ("BIT", "bit-2565", "2565", "no_coop"),
            ("AIT", "ait-2566", "2566", "default"),
            ("GENED", "gened-2557", "2557", "gened"),
        ]
        paths = []
        for index, (program, catalog_key, academic_year, plan) in enumerate(editions):
            path = directory / f"curriculum-{index:02d}.json"
            path.write_text(
                json.dumps(
                    {
                        "program": program,
                        "plan": plan,
                        "catalog": {
                            "catalog_key": catalog_key,
                            "academic_year": academic_year,
                        },
                        "courses": [{"code": f"C{index:03d}"}],
                        "source_provenance": [{
                            "program": program,
                            "source_filename": f"{catalog_key}-{plan}.pdf",
                            "source_page": 1,
                            "document_category": "plan",
                        }],
                    }
                ),
                encoding="utf-8",
            )
            paths.append(path)
        return paths

    def test_real_canonical_documents_load_without_curriculum_masquerade(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "curriculum.db"
            load_jsons_to_sqlite(self._canonical_curricula(root), database)
            load_supplemental_jsons_to_sqlite(self.policy, self.requirements, database)

            with closing(sqlite3.connect(database)) as connection:
                counts = connection.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM regulation_rules),
                        (SELECT COUNT(*) FROM policy_facts),
                        (SELECT COUNT(*) FROM program_requirements),
                        (SELECT COUNT(*) FROM courses)
                    """
                ).fetchone()
                totals = connection.execute(
                    """
                    SELECT catalogs.catalog_key, program_code, value
                    FROM program_requirements
                    JOIN catalogs USING (catalog_id)
                    ORDER BY catalogs.catalog_key
                    """
                ).fetchall()
                registration = connection.execute(
                    """
                    SELECT value, operator, source_rule_id
                    FROM policy_facts
                    WHERE fact_key = 'regular_maximum'
                    """
                ).fetchone()
                provenance = connection.execute(
                    """
                    SELECT provenance.document_category, provenance.source_page
                    FROM policy_fact_provenance AS links
                    JOIN provenance
                      ON provenance.provenance_id = links.provenance_id
                    JOIN policy_facts
                      ON policy_facts.fact_id = links.fact_id
                    WHERE policy_facts.fact_key = 'regular_maximum'
                    """
                ).fetchone()

            self.assertEqual(counts[2:], (5, 10))
            self.assertEqual(
                totals,
                [
                    ("ait-2566", "AIT", 120),
                    ("bit-2565", "BIT", 126),
                    ("dsba-2560", "DSBA", 126),
                    ("dsba-2565", "DSBA", 132),
                    ("it-2565", "IT", 129),
                ],
            )
            self.assertEqual(registration, (22, "<=", "rule:11"))
            self.assertEqual(provenance[0], "rule")
            self.assertIsNotNone(provenance[1])

    def test_unsupported_supplemental_provenance_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "source": "Academic Rules",
                        "categories": [
                            {
                                "category": "test",
                                "values": [
                                    {
                                        "value": "1",
                                        "condition": "at_least",
                                        "source_rule_id": "rule:1",
                                    }
                                ],
                                "evidence": {
                                    "source_provenance": [
                                        {
                                            "source_filename": "rule.png",
                                            "source_page": 1,
                                            "document_category": "unsupported",
                                        }
                                    ],
                                    "supporting_rule_text": [
                                        {"rule_id": "rule:1", "rule_text": "rule"}
                                    ],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            requirements = root / "requirements.json"
            requirements.write_text("[]", encoding="utf-8")
            database = root / "curriculum.db"
            load_json_to_sqlite(self._curriculum(root), database)

            with self.assertRaisesRegex(ValueError, "unsupported document category"):
                load_supplemental_jsons_to_sqlite(policy, requirements, database)

    def test_rule_json_cannot_be_loaded_as_curriculum(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "curriculum.db"
            with self.assertRaisesRegex(ValueError, "courses list"):
                load_json_to_sqlite(self.policy, database)

    def test_repeated_loads_have_deterministic_authority_rows(self):
        rows = []
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory in (Path(first), Path(second)):
                database = directory / "curriculum.db"
                load_jsons_to_sqlite(self._canonical_curricula(directory), database)
                load_supplemental_jsons_to_sqlite(self.policy, self.requirements, database)
                with closing(sqlite3.connect(database)) as connection:
                    rows.append(
                        (
                            connection.execute(
                                "SELECT regulation_rules.rule_id, regulation_rules.category, "
                                "policy_facts.source_rule_id FROM "
                                "regulation_rules LEFT JOIN policy_facts "
                                "ON policy_facts.source_rule_id = regulation_rules.rule_id "
                                "ORDER BY rule_id, fact_id"
                            ).fetchall(),
                            connection.execute(
                                "SELECT catalogs.catalog_key, program_code, "
                                "requirement_type, value FROM program_requirements "
                                "JOIN catalogs USING (catalog_id) "
                                "ORDER BY catalogs.catalog_key"
                            ).fetchall(),
                        )
                    )
        self.assertEqual(rows[0], rows[1])

    def test_index_build_loads_supplemental_inputs_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = self._canonical_curricula(root)
            artifact_dir = root / "runtime"
            embed_calls = []

            def fake_embed(texts):
                values = list(texts)
                embed_calls.append(values)
                return [[0.0] * 384 for _ in values]

            database = ensure_index(
                sources,
                artifact_dir=artifact_dir,
                supplemental_json_paths={
                    "institution_policy": self.policy,
                    "program_requirements": self.requirements,
                },
                embed_texts_callable=fake_embed,
                embedding_model_identity="r4-test",
            )
            with closing(sqlite3.connect(database)) as connection:
                sources = connection.execute(
                    "SELECT source_filename FROM semantic_index_sources "
                    "ORDER BY source_filename"
                ).fetchall()
                sources = [row[0] for row in sources]
                rule_count = connection.execute(
                    "SELECT COUNT(*) FROM regulation_rules"
                ).fetchone()[0]
                requirement_count = connection.execute(
                    "SELECT COUNT(*) FROM program_requirements"
                ).fetchone()[0]
            self.assertEqual(
                sources,
                [
                    *(f"curriculum-{index:02d}.json" for index in range(10)),
                    "institution_policy.json",
                    "program_requirements.json",
                ],
            )
            self.assertGreater(rule_count, 0)
            self.assertEqual(requirement_count, 5)
            self.assertEqual(len(embed_calls), 1)


if __name__ == "__main__":
    unittest.main()
