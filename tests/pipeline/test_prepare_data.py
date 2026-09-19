import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipeline.tools.preparation import tool as prepare_data


def _touch_pages(directory: Path, pages: range, prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for page in pages:
        (directory / f"{prefix}_page_{page:03d}_ocr.json").write_text(
            "{}", encoding="utf-8"
        )


class PrepareDataTests(unittest.TestCase):
    def test_explicit_scope_configuration_matches_current_pipeline(self):
        self.assertEqual(prepare_data.PROGRAM_CONFIG["it"].scopes[0].pages, "32-38")
        self.assertEqual(prepare_data.PROGRAM_CONFIG["it"].scopes[1].pages, "39-45")
        self.assertEqual(
            prepare_data.PROGRAM_CONFIG["it"].shared_description.pages,
            "328-371",
        )
        self.assertEqual(prepare_data.PROGRAM_CONFIG["gened"].scopes[0].pages, "16-30,44-117")

    def test_partial_corpus_skips_missing_programs_and_unknown_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ocr = root / "data" / "output" / "ocr"
            _touch_pages(ocr / "it", range(32, 39), "it")
            (ocr / "bit").mkdir(parents=True)
            (ocr / "custom").mkdir()

            with patch("src.pipeline.tools.preparation.tool._run_extract") as run_extract, patch(
                "src.pipeline.tools.preparation.tool._run_merge"
            ) as run_merge:
                result = prepare_data.prepare_data(root)

            self.assertEqual(result["prepared_programs"], ["it"])
            self.assertIn("bit", result["skipped_programs"])
            self.assertEqual(result["unknown_directories"], ["custom"])
            self.assertEqual(run_extract.call_count, 1)
            self.assertEqual(run_merge.call_count, 1)

    def test_shared_description_range_is_extracted_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ocr = root / "data" / "output" / "ocr" / "it"
            _touch_pages(ocr, range(32, 39), "it")
            _touch_pages(ocr, range(328, 372), "it")

            with patch("src.pipeline.tools.preparation.tool._run_extract") as run_extract, patch(
                "src.pipeline.tools.preparation.tool._run_merge"
            ) as run_merge:
                prepare_data.prepare_data(root)

            self.assertEqual(run_extract.call_count, 2)
            extracted_scopes = [call.args[4] for call in run_extract.call_args_list]
            self.assertEqual(
                [scope.pages for scope in extracted_scopes], ["32-38", "328-371"]
            )
            self.assertEqual(run_merge.call_count, 1)
            self.assertEqual(run_merge.call_args.args[4], "32-38,328-371")

    def test_missing_scope_pages_are_omitted_cleanly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _touch_pages(root / "data" / "output" / "ocr" / "it", range(39, 46), "it")

            with patch("src.pipeline.tools.preparation.tool._run_extract") as run_extract, patch(
                "src.pipeline.tools.preparation.tool._run_merge"
            ) as run_merge:
                result = prepare_data.prepare_data(root)

            self.assertEqual(result["prepared_programs"], ["it"])
            self.assertEqual(run_extract.call_count, 1)
            self.assertEqual(run_extract.call_args.args[4].plan, "coop")
            self.assertEqual(run_merge.call_count, 1)

    def test_empty_or_unknown_only_corpus_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "output" / "ocr" / "it").mkdir(parents=True)
            (root / "data" / "output" / "ocr" / "other").mkdir()

            with self.assertRaisesRegex(
                prepare_data.PreparationError, "No usable supported OCR source"
            ):
                prepare_data.prepare_data(root)

        with patch.object(
            prepare_data,
            "prepare_data",
            side_effect=prepare_data.PreparationError("No usable supported OCR source"),
        ):
            self.assertEqual(prepare_data.main(), 1)

    def test_only_existing_stage_commands_are_orchestrated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _touch_pages(root / "data" / "output" / "ocr" / "ait", range(1, 2), "ait")

            with patch("src.pipeline.tools.preparation.tool._run_extract") as run_extract, patch(
                "src.pipeline.tools.preparation.tool._run_merge"
            ) as run_merge:
                prepare_data.prepare_data(root)

            for call in run_extract.call_args_list:
                self.assertNotIn("run_pipeline", repr(call))
                self.assertNotIn("llm_spell_corrector", repr(call))
                self.assertNotIn("src.pipeline.tools.evaluation.evaluate.py", repr(call))
                self.assertNotIn("rag", repr(call))
            self.assertEqual(run_extract.call_count, 1)
            self.assertEqual(run_merge.call_count, 1)


if __name__ == "__main__":
    unittest.main()
