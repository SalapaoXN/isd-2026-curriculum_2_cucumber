import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stderr
from io import StringIO
from pathlib import Path

from rag.structured.loader import load_json_to_sqlite
from src.pipeline.run import (
    _apply_edition_metadata,
    _edition_metadata_from_args,
    main,
    parse_args,
)
from src.pipeline.tools.correction.corrector import correct_json_files


class EditionMetadataTests(unittest.TestCase):
    def test_run_configuration_carries_explicit_catalog_metadata(self):
        args = parse_args(
            [
                "--program",
                "dsba",
                "--catalog-key",
                "dsba-2568-coop",
                "--academic-year",
                "2568",
            ]
        )

        self.assertEqual(
            _edition_metadata_from_args(args),
            {"catalog_key": "dsba-2568-coop", "academic_year": "2568"},
        )

    def test_explicit_metadata_reaches_corrected_loader_compatible_artifact(self):
        merged_document = {
            "program": "DSBA",
            "plan": "coop",
            "courses": [{"code": "C100"}],
        }
        metadata = {"catalog_key": "dsba-2568-coop", "academic_year": "2568"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_path = root / "merged_dsba_coop_full.json"
            merged_path.write_text(json.dumps(merged_document), encoding="utf-8")
            merged_path = _apply_edition_metadata([merged_path], metadata)[0]

            corrected_path = correct_json_files([merged_path], root / "final")[0]
            corrected_document = json.loads(corrected_path.read_text(encoding="utf-8"))
            database_path = root / "curriculum.db"
            load_json_to_sqlite(corrected_path, database_path)
            with closing(sqlite3.connect(database_path)) as connection:
                stored_catalog = connection.execute(
                    "SELECT catalog_key, academic_year FROM catalogs"
                ).fetchone()

        self.assertEqual(corrected_document["program"], "DSBA")
        self.assertEqual(corrected_document["plan"], "coop")
        self.assertEqual(
            corrected_document["catalog"],
            {"catalog_key": "dsba-2568-coop", "academic_year": "2568"},
        )
        self.assertEqual(stored_catalog, ("dsba-2568-coop", "2568"))

    def test_same_program_and_plan_keep_distinct_explicit_catalog_identities(self):
        metadata_by_year = [
            {"catalog_key": "dsba-2565-coop", "academic_year": "2565"},
            {"catalog_key": "dsba-2568-coop", "academic_year": "2568"},
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for year, metadata in zip(("2565", "2568"), metadata_by_year):
                path = root / f"dsba_coop_{year}.json"
                path.write_text(
                    json.dumps({"program": "DSBA", "plan": "coop", "courses": []}),
                    encoding="utf-8",
                )
                path = _apply_edition_metadata([path], metadata)[0]
                paths.append(json.loads(path.read_text(encoding="utf-8")))

        self.assertEqual([(doc["program"], doc["plan"]) for doc in paths], [("DSBA", "coop")] * 2)
        self.assertEqual(
            [doc["catalog"]["catalog_key"] for doc in paths],
            ["dsba-2565-coop", "dsba-2568-coop"],
        )
        self.assertEqual(
            [doc["catalog"]["academic_year"] for doc in paths], ["2565", "2568"]
        )

    def test_filename_does_not_supply_missing_edition_year(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dsba_9999_coop.json"
            path.write_text(
                json.dumps({"program": "DSBA", "plan": "coop", "courses": []}),
                encoding="utf-8",
            )
            _apply_edition_metadata([path], None)
            document = json.loads(path.read_text(encoding="utf-8"))

        self.assertNotIn("catalog", document)

    def test_malformed_explicit_academic_year_is_rejected(self):
        for academic_year in ("", "abc", "25", True, [2568], {"year": 2568}):
            with self.subTest(academic_year=academic_year):
                args = parse_args(
                    ["--catalog-key", "dsba-test", "--academic-year", "2568"]
                )
                args.academic_year = academic_year
                with self.assertRaisesRegex(ValueError, "academic_year"):
                    _edition_metadata_from_args(args)

    def test_cli_rejects_malformed_year_before_running_pipeline(self):
        errors = StringIO()

        with redirect_stderr(errors):
            result = main(
                ["--catalog-key", "dsba-test", "--academic-year", "not-a-year"]
            )

        self.assertEqual(result, 2)
        self.assertIn("Invalid edition metadata: academic_year", errors.getvalue())

    def test_legacy_configuration_without_edition_metadata_is_optional(self):
        args = parse_args(["--program", "dsba"])

        self.assertIsNone(_edition_metadata_from_args(args))


if __name__ == "__main__":
    unittest.main()
