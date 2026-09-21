import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipeline import run as pipeline_run
from src.pipeline.tools.indexing import tool as indexing_tool


class IndexingSafetyTests(unittest.TestCase):
    @staticmethod
    def _write_source(directory: Path, program: str) -> Path:
        path = directory / f"{program.lower()}_corrected.json"
        path.write_text(json.dumps({"program": program, "courses": []}), encoding="utf-8")
        return path

    def test_shared_runtime_rejects_scoped_sources_before_build(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_source(root, "IT")
            artifact_dir = root / "cucumber_outputs" / "runtime"
            artifact_dir.mkdir(parents=True)
            database = artifact_dir / indexing_tool.DEFAULT_INDEX_NAME
            sentinel = database.with_suffix(".h1-test-sentinel")
            sentinel.write_text("must remain untouched", encoding="utf-8")
            try:
                with patch.object(indexing_tool, "ARTIFACTS_DIR", artifact_dir):
                    with patch.object(indexing_tool, "build_index") as build:
                        with self.assertRaisesRegex(ValueError, "incomplete source set"):
                            indexing_tool.run_build_index_stage([source])
                build.assert_not_called()
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "must remain untouched")
            finally:
                sentinel.unlink(missing_ok=True)

    def test_complete_source_set_is_accepted_for_shared_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = [self._write_source(root, program) for program in indexing_tool._UNIFIED_PROGRAMS]
            artifact_dir = root / "cucumber_outputs" / "runtime"
            expected_path = artifact_dir / indexing_tool.DEFAULT_INDEX_NAME
            with patch.object(indexing_tool, "ARTIFACTS_DIR", artifact_dir):
                with patch.object(indexing_tool, "build_index", return_value=expected_path) as build:
                    result = indexing_tool.run_build_index_stage(sources)

            self.assertEqual(result, expected_path)
            build.assert_called_once_with(sources, None)

    def test_shared_runtime_validation_preserves_iterable_sources_for_build(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = [self._write_source(root, program) for program in indexing_tool._UNIFIED_PROGRAMS]
            artifact_dir = root / "cucumber_outputs" / "runtime"
            expected_path = artifact_dir / indexing_tool.DEFAULT_INDEX_NAME
            with patch.object(indexing_tool, "ARTIFACTS_DIR", artifact_dir):
                with patch.object(indexing_tool, "build_index", return_value=expected_path) as build:
                    result = indexing_tool.run_build_index_stage(iter(sources))

            self.assertEqual(result, expected_path)
            build.assert_called_once_with(sources, None)

    def test_scoped_source_set_requires_a_distinct_index_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_source(root, "IT")
            artifact_dir = root / "cucumber_outputs" / "runtime"
            scoped_index = artifact_dir / "h1-scoped-test.db"
            with patch.object(indexing_tool, "ARTIFACTS_DIR", artifact_dir):
                with patch.object(indexing_tool, "build_index", return_value=scoped_index) as build:
                    result = indexing_tool.run_build_index_stage([source], scoped_index)

            self.assertEqual(result, scoped_index)
            build.assert_called_once_with([source], scoped_index)

    def test_only_index_default_target_uses_unified_source_discovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            with patch(
                "src.pipeline.tools.indexing.tool.run_build_index_stage"
            ) as build:
                result = pipeline_run.main(
                    [
                        "--program",
                        "it",
                        "--only-index",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(result, 0)
            build.assert_called_once_with(None, None)

    def test_with_index_default_target_uses_unified_source_discovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            with patch(
                "src.pipeline.tools.indexing.tool.run_build_index_stage"
            ) as build:
                result = pipeline_run.main(
                    [
                        "--program",
                        "it",
                        "--from",
                        "corrected",
                        "--skip-eval",
                        "--with-index",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(result, 0)
            build.assert_called_once_with(None, None)


if __name__ == "__main__":
    unittest.main()
