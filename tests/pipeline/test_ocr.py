import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.pipeline.tools.ocr import cli as ocr
from src.pipeline.tools.ocr.pipeline_runner import run_ocr


class OcrCliTests(unittest.TestCase):
    def _run_wrapper(self, root: Path, *arguments: str) -> None:
        with patch.object(ocr, "BASE_DIR", root), patch.object(ocr, "run_ocr"):
            with patch.object(sys, "argv", ["ocr.py", *arguments]):
                ocr.main()

    def test_supported_prefixes_map_to_inputs_and_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for prefix in ocr.SUPPORTED_PREFIXES:
                (root / "data" / "input" / prefix).mkdir(parents=True)
                with patch.object(ocr, "BASE_DIR", root), patch.object(
                    ocr, "run_ocr"
                ) as run_mock:
                    with patch.object(sys, "argv", ["ocr.py", "--prefix", prefix]):
                        ocr.main()

                run_mock.assert_called_once_with(
                    input_dir=root / "data" / "input" / prefix,
                    output_dir=root / "data" / "output",
                    program=prefix.upper(),
                    pages=None,
                    no_gpu=False,
                )

    def test_page_range_is_forwarded_and_no_gpu_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "input" / "it").mkdir(parents=True)
            with patch.object(ocr, "BASE_DIR", root), patch.object(
                ocr, "run_ocr"
            ) as run_mock:
                with patch.object(
                    sys,
                    "argv",
                    ["ocr.py", "--prefix", "it", "--pages", "32-38", "--no-gpu"],
                ):
                    ocr.main()

            run_mock.assert_called_once_with(
                input_dir=root / "data" / "input" / "it",
                output_dir=root / "data" / "output",
                program="IT",
                pages=list(range(32, 39)),
                no_gpu=True,
            )

    def test_unsupported_prefix_is_rejected(self):
        with patch.object(sys, "argv", ["ocr.py", "--prefix", "unknown"]):
            with self.assertRaisesRegex(SystemExit, "Unsupported prefix"):
                ocr.main()

    def test_missing_input_directory_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(ocr, "BASE_DIR", Path(temp_dir)):
                with patch.object(sys, "argv", ["ocr.py", "--prefix", "it"]):
                    with self.assertRaisesRegex(
                        SystemExit, "does not exist or is not a directory"
                    ):
                        ocr.main()

    def test_empty_input_directory_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "input" / "it").mkdir(parents=True)
            with patch.object(ocr, "BASE_DIR", root):
                with patch.object(sys, "argv", ["ocr.py", "--prefix", "it"]):
                    with self.assertRaisesRegex(SystemExit, "No valid page images"):
                        ocr.main()

    def test_ocr_writes_only_ocr_boundary_and_processes_all_pages_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "data" / "input" / "it"
            output_dir = root / "data" / "output"
            input_dir.mkdir(parents=True)
            for page in (1, 3):
                (input_dir / f"it_page_{page:03d}.jpg").write_bytes(b"fixture")

            class FakeOCREngine:
                seen_pages = []

                def __init__(self, **_kwargs):
                    pass

                def extract_text(self, image, detail=0):
                    self.seen_pages.append(image.name)
                    self.detail = detail
                    return ["  OCR LINE  "]

            with patch("src.pipeline.tools.ocr.pipeline_runner.OCREngine", FakeOCREngine):
                with redirect_stdout(io.StringIO()):
                    result = run_ocr(
                        input_dir=input_dir,
                        output_dir=output_dir,
                        program="IT",
                        pages=None,
                        no_gpu=True,
                    )

            self.assertEqual(result, output_dir / "ocr" / "it")
            self.assertEqual(FakeOCREngine.seen_pages, ["it_page_001.jpg", "it_page_003.jpg"])
            self.assertEqual(
                json.loads(
                    (output_dir / "ocr" / "it" / "it_page_001_ocr.json").read_text(
                        encoding="utf-8"
                    )
                )["text_lines"],
                ["OCR LINE"],
            )
            self.assertFalse((output_dir / "extracted").exists())
            self.assertFalse((output_dir / "consolidated").exists())


if __name__ == "__main__":
    unittest.main()
