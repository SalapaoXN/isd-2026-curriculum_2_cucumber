import json
import tempfile
import unittest
from pathlib import Path

from extract import _filter_files_by_prefix
from merge_consecutive import merge_consecutive_files
from src.run_pipeline import parse_pages
from src.pipeline_config import (
    discover_page_files,
    discover_pages,
    resolve_plan,
    resolve_program,
)

class CliSemanticsTests(unittest.TestCase):
    def test_page_parser_supports_ranges_and_rejects_invalid_values(self):
        self.assertEqual(parse_pages("10-12,12,14..15"), [10, 11, 12, 14, 15])

        with self.assertRaisesRegex(ValueError, "Invalid page value"):
            parse_pages("10,wrong")

    def test_page_discovery_is_direct_numeric_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "ait"
            input_dir.mkdir()
            (input_dir / "ait_page_010.png").write_bytes(b"")
            (input_dir / "ait_page_002.webp").write_bytes(b"")
            (input_dir / "ait_page_002.jpg").write_bytes(b"")
            (input_dir / "other_page_001.png").write_bytes(b"")
            nested = input_dir / "nested"
            nested.mkdir()
            (nested / "ait_page_001.png").write_bytes(b"")

            self.assertEqual(discover_pages(input_dir), [2, 10])
            self.assertEqual(discover_page_files(input_dir)[2].suffix, ".jpg")

    def test_page_discovery_fails_without_valid_direct_children(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "ait"
            input_dir.mkdir()
            (input_dir / "ait_page_01.png").write_bytes(b"")

            with self.assertRaisesRegex(ValueError, "No valid page images"):
                discover_pages(input_dir)

    def test_program_is_derived_only_from_supported_directory_names(self):
        self.assertEqual(resolve_program(None, Path("inputs/ait")), "AIT")
        self.assertEqual(resolve_program("it", Path("inputs/custom")), "IT")

        with self.assertRaisesRegex(ValueError, "Cannot derive a program"):
            resolve_program(None, Path("inputs/custom"))

    def test_plan_resolution_does_not_guess_variants(self):
        self.assertEqual(resolve_plan("no-coop", "DSBA"), "no_coop")
        self.assertEqual(resolve_plan(None, "AIT"), None)
        self.assertEqual(resolve_plan("gened", "GENED"), "gened")

        with self.assertRaisesRegex(ValueError, "explicit --plan"):
            resolve_plan(None, "IT")
        with self.assertRaisesRegex(ValueError, "AIT has no study-plan"):
            resolve_plan("coop", "AIT")
        with self.assertRaisesRegex(ValueError, "GENED requires"):
            resolve_plan(None, "GENED")

    def test_ait_merge_keeps_null_plan_and_uses_neutral_filename_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "outputs"
            output_dir = Path(temp_dir) / "consolidated"
            input_dir.mkdir()
            (input_dir / "ait_page_023_ocr_extracted.json").write_text(
                json.dumps(
                    {
                        "program": "AIT",
                        "plan": None,
                        "courses": [{"code": "06000001"}],
                    }
                ),
                encoding="utf-8",
            )

            merge_consecutive_files(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                prefix="ait",
            )

            merged_files = list(output_dir.glob("*.json"))
            self.assertEqual(len(merged_files), 1)
            self.assertIn("no_plan", merged_files[0].name)
            result = json.loads(merged_files[0].read_text(encoding="utf-8"))
            self.assertIsNone(result["plan"])
    
    def test_extract_prefix_filters_only_matching_ocr_files(self):
        files = [
            Path("outputs/gened_page_016_ocr.json"),
            Path("outputs/gened_page_044_ocr.json"),
            Path("outputs/dsba_page_026_ocr.json"),
            Path("outputs/it_page_032_ocr.json"),
        ]

        result = _filter_files_by_prefix(files, "gened")

        self.assertEqual(
            [file.name for file in result],
            [
                "gened_page_016_ocr.json",
                "gened_page_044_ocr.json",
            ],
        )
        self.assertEqual(
            _filter_files_by_prefix(
                [Path("outputs/GENED_page_016_ocr.json")],
                "gened",
            ),
            [Path("outputs/GENED_page_016_ocr.json")],
        )


if __name__ == "__main__":
    unittest.main()
