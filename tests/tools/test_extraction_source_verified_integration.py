import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipeline.tools.extraction import tool as extraction_tool


class SourceVerifiedExtractionIntegrationTests(unittest.TestCase):
    @staticmethod
    def _course(output_root: Path, program: str, code: str) -> dict:
        matches = []
        for path in output_root.rglob("*_extracted.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            matches.extend(
                course
                for course in payload.get("courses", [])
                if course.get("code") == code
            )
        if len(matches) != 1:
            raise AssertionError(f"expected one {program} {code}, found {len(matches)}")
        return matches[0]

    def test_run_extraction_applies_reviewed_it_credit_and_preserves_provenance(self):
        source = Path("outputs/ocr/it/it_page_354_ocr.json")
        self.assertEqual(
            extraction_tool.SOURCE_VERIFIED_CREDIT_CORRECTIONS_PATH,
            Path("data/corrections/source_verified_credit_corrections.json").resolve(),
        )
        self.assertTrue(extraction_tool.SOURCE_VERIFIED_CREDIT_CORRECTIONS_PATH.is_file())

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                extraction_tool,
                "_load_source_verified_credit_corrections",
                wraps=extraction_tool._load_source_verified_credit_corrections,
            ) as load_corrections:
                extraction_tool.run_extraction(
                    source,
                    Path(temp_dir),
                    program="IT",
                    plan="coop",
                )

            load_corrections.assert_called_once_with()
            repaired = self._course(Path(temp_dir), "IT", "06016454")
            self.assertEqual(repaired["credits"], "3(3-0-6)")
            self.assertIs(repaired["credit_source_verified"], True)
            self.assertEqual(
                repaired["credit_source_provenance"],
                {
                    "source_filename": "it_page_354.png",
                    "source_page": 354,
                    "document_category": "description",
                },
            )
            source_entry = repaired["source_provenance"][0]
            self.assertEqual(source_entry["source_filename"], "it_page_354.png")
            self.assertEqual(source_entry["source_page"], 354)
            self.assertEqual(source_entry["document_category"], "description")
            self.assertIs(source_entry["source_verified"], True)

            existing = self._course(Path(temp_dir), "IT", "06016453")
            self.assertEqual(existing["credits"], "3(3-0-6)")
            self.assertNotIn("credit_source_verified", existing)

    def test_run_extraction_leaves_bit_credit_unresolved(self):
        source = Path("outputs/ocr/bit/bit_page_252_ocr.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            extraction_tool.run_extraction(
                source,
                Path(temp_dir),
                program="BIT",
                plan="coop",
            )
            unresolved = self._course(Path(temp_dir), "BIT", "06036135")
            self.assertNotEqual(unresolved.get("credits"), "3(3-0-6)")
            self.assertNotIn("credit_source_verified", unresolved)

            existing = self._course(Path(temp_dir), "BIT", "06036134")
            self.assertEqual(existing["credits"], "3(2-2-5)")

    def test_run_extraction_has_no_ground_truth_dependency(self):
        source = inspect.getsource(extraction_tool.run_extraction)
        self.assertNotIn("ground_truth", source.lower())
        self.assertNotIn("_load_authoritative_credit_lookup", source)

    def test_malformed_correction_artifact_fails_before_extraction(self):
        source = Path("outputs/ocr/it/it_page_354_ocr.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            correction_path = Path(temp_dir) / "malformed.json"
            correction_path.write_text(
                json.dumps({"corrections": [{"program": "IT"}]}),
                encoding="utf-8",
            )
            with patch.object(
                extraction_tool,
                "SOURCE_VERIFIED_CREDIT_CORRECTIONS_PATH",
                correction_path,
            ):
                with self.assertRaises(ValueError):
                    extraction_tool.run_extraction(source, Path(temp_dir) / "output", "IT", "coop")

    def test_mismatched_correction_identity_fails_closed(self):
        source = Path("outputs/ocr/it/it_page_354_ocr.json")
        mismatched = {
            "version": 1,
            "corrections": [
                {
                    "program": "IT",
                    "plan": "coop",
                    "course_code": "06016454",
                    "credits": "3(3-0-6)",
                    "source_verified": True,
                    "source_filename": "it_page_355.png",
                    "source_page": 355,
                    "document_category": "description",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            correction_path = Path(temp_dir) / "mismatched.json"
            correction_path.write_text(json.dumps(mismatched), encoding="utf-8")
            with patch.object(
                extraction_tool,
                "SOURCE_VERIFIED_CREDIT_CORRECTIONS_PATH",
                correction_path,
            ):
                extraction_tool.run_extraction(
                    source,
                    Path(temp_dir) / "output",
                    program="IT",
                    plan="coop",
                )
            unresolved = self._course(Path(temp_dir) / "output", "IT", "06016454")
            self.assertNotEqual(unresolved.get("credits"), "3(3-0-6)")
            self.assertNotIn("credit_source_verified", unresolved)


if __name__ == "__main__":
    unittest.main()
