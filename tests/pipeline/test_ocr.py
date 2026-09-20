import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.pipeline.tools.ocr import cli as ocr
from src.pipeline.tools.extraction.engine import CurriculumExtractor
from src.pipeline.utils.file_handler import save_ocr_results
from src.pipeline.tools.ocr.engine import OCREngine
from src.pipeline.utils.page_metadata import resolve_document_page
from src.pipeline.tools.extraction.rules import RulePage
from src.pipeline.tools.ocr.pipeline_runner import _document_page_for_ocr, run_ocr


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
                details = []

                def __init__(self, **_kwargs):
                    pass

                def extract_text(self, image, detail=0):
                    self.seen_pages.append(image.name)
                    self.details.append(detail)
                    return [
                        {
                            "text": "  OCR LINE  ",
                            "confidence": 0.91,
                            "bbox": [[1, 2], [3, 2], [3, 4], [1, 4]],
                        }
                    ]

            class FakeImage:
                size = (640, 480)

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

            with patch(
                "src.pipeline.tools.ocr.pipeline_runner.OCREngine", FakeOCREngine
            ), patch(
                "src.pipeline.tools.ocr.pipeline_runner.Image.open",
                return_value=FakeImage(),
            ):
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
            self.assertEqual(FakeOCREngine.details, [1, 1])
            raw = json.loads(
                (output_dir / "ocr" / "it" / "it_page_001_ocr.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                raw["text_lines"],
                ["OCR LINE"],
            )
            self.assertEqual(raw["detections"][0]["text"], "  OCR LINE  ")
            self.assertEqual(raw["image_width"], 640)
            self.assertEqual(raw["image_height"], 480)
            self.assertEqual(raw["source_filename"], "it_page_001.jpg")
            self.assertEqual(raw["source_page"], 1)
            self.assertEqual(raw["program"], "IT")
            self.assertFalse((output_dir / "extracted").exists())
            self.assertFalse((output_dir / "consolidated").exists())

    def test_detail_one_normalization_preserves_order_and_detail_zero_api(self):
        class NumpyLike:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class FakeReader:
            def readtext(self, _path, detail=0):
                if detail == 0:
                    return ["first", "second"]
                return [
                    (
                        [
                            [NumpyLike(10), NumpyLike(20)],
                            [NumpyLike(30), NumpyLike(20)],
                            [NumpyLike(30), NumpyLike(40)],
                            [NumpyLike(10), NumpyLike(40)],
                        ],
                        "second",
                        NumpyLike(0.82),
                    ),
                    (
                        [
                            [NumpyLike(1), NumpyLike(2)],
                            [NumpyLike(3), NumpyLike(2)],
                            [NumpyLike(3), NumpyLike(4)],
                            [NumpyLike(1), NumpyLike(4)],
                        ],
                        "first",
                        NumpyLike(0.91),
                    ),
                ]

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image_path.write_bytes(b"fixture")
            engine = object.__new__(OCREngine)
            engine.reader = FakeReader()

            self.assertEqual(engine.extract_text(image_path, detail=0), ["FIRST", "SECOND"])
            detections = engine.extract_text(image_path, detail=1)

        self.assertEqual([item["text"] for item in detections], ["second", "first"])
        self.assertEqual(detections[0]["confidence"], 0.82)
        self.assertEqual(detections[0]["bbox"][0], [10, 20])

    def test_ocr_serialization_is_optional_and_consumers_ignore_geometry(self):
        class NumpyLike:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        lines = ["ปีที่ 1", "06000001", "ชื่อวิชา", "3(3-0-6)"]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            save_ocr_results(
                lines,
                output_dir,
                "legacy",
                source_filename="it_page_032.png",
                source_page=32,
                program="IT",
            )
            save_ocr_results(
                lines,
                output_dir,
                "geometry",
                source_filename="it_page_032.png",
                source_page=32,
                program="IT",
                detections=[
                    {
                        "text": "06000001",
                        "confidence": NumpyLike(0.99),
                        "bbox": [[NumpyLike(1), NumpyLike(2)]] * 4,
                    }
                ],
                image_width=1200,
                image_height=1600,
            )

            legacy_path = output_dir / "legacy_ocr.json"
            geometry_path = output_dir / "geometry_ocr.json"
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
            legacy_result = CurriculumExtractor(program="IT", plan="no_coop").process_file(
                legacy_path
            )
            geometry_result = CurriculumExtractor(program="IT", plan="no_coop").process_file(
                geometry_path
            )
            legacy_rule_page = RulePage.from_file(legacy_path)
            geometry_rule_page = RulePage.from_file(geometry_path)

        self.assertNotIn("detections", legacy)
        self.assertNotIn("image_width", legacy)
        self.assertNotIn("image_height", legacy)
        self.assertEqual(geometry["detections"][0]["confidence"], 0.99)
        self.assertEqual(geometry["detections"][0]["bbox"][0], [1, 2])
        self.assertEqual(geometry["image_width"], 1200)
        self.assertEqual(geometry["image_height"], 1600)
        self.assertEqual(legacy_result["courses"][0]["code"], "06000001")
        self.assertEqual(geometry_result["courses"][0]["code"], "06000001")
        self.assertEqual(list(legacy_rule_page.lines), lines)
        self.assertEqual(list(geometry_rule_page.lines), lines)

    def test_document_page_propagation_preserves_detector_rules_and_explicit_page(self):
        cases = {
            "top": (["27", "header", "content", "course", "credits", "tail"], 27),
            "bottom": (["header", "content", "course", "credits", "tail", "27"], 27),
            "ambiguous": (["27", "header", "content", "course", "28", "tail"], None),
            "outside": (["header0", "header1", "header2", "header3", "header4", "27",
                         "tail6", "tail7", "tail8", "tail9", "tail10"], None),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            for name, (lines, expected_page) in cases.items():
                save_ocr_results(
                    lines,
                    output_dir,
                    name,
                    source_filename="it_page_032.png",
                    source_page=32,
                    program="IT",
                )

            save_ocr_results(
                cases["ambiguous"][0],
                output_dir,
                "explicit",
                source_filename="it_page_032.png",
                source_page=32,
                program="IT",
                document_page=99,
            )

            results = {
                name: json.loads(
                    (output_dir / f"{name}_ocr.json").read_text(encoding="utf-8")
                )
                for name in [*cases, "explicit"]
            }

        for name, (_, expected_page) in cases.items():
            self.assertEqual(results[name]["document_page"], expected_page)
            self.assertEqual(results[name]["source_page"], 32)
            self.assertEqual(results[name]["program"], "IT")
            self.assertEqual(results[name]["text_lines"], cases[name][0])
        self.assertEqual(results["explicit"]["document_page"], 99)
        self.assertEqual(results["explicit"]["source_page"], 32)
        self.assertEqual(results["explicit"]["text_lines"], cases["ambiguous"][0])

    def test_gened_bounded_document_page_fallback_is_strictly_scoped(self):
        unresolved_lines = ["header", "content", "course", "credits", "tail"]
        self.assertEqual(_document_page_for_ocr(unresolved_lines, "GENED", 44), 40)
        self.assertEqual(_document_page_for_ocr(unresolved_lines, "GENED", 74), 70)
        self.assertEqual(_document_page_for_ocr(unresolved_lines, "GENED", 81), 77)
        self.assertEqual(_document_page_for_ocr(unresolved_lines, "GENED", 104), 100)
        self.assertEqual(_document_page_for_ocr(unresolved_lines, "GENED", 117), 113)
        self.assertIsNone(_document_page_for_ocr(unresolved_lines, "GENED", 43))
        self.assertIsNone(_document_page_for_ocr(unresolved_lines, "GENED", 118))
        self.assertIsNone(_document_page_for_ocr(unresolved_lines, "AIT", 74))
        self.assertIsNone(_document_page_for_ocr(unresolved_lines, "DSBA", 81))
        self.assertIsNone(_document_page_for_ocr(unresolved_lines, "IT", 104))
        self.assertEqual(_document_page_for_ocr(["70", "header"], "GENED", 74), 70)
        self.assertEqual(
            _document_page_for_ocr(["70", "header", "71"], "GENED", 74), 70
        )

    def test_document_page_resolver_uses_exact_source_verified_override(self):
        result = resolve_document_page(
            "DSBA", 28, "dsba_page_028.png", ["header", "content"]
        )
        self.assertEqual(result.document_page, 23)
        self.assertEqual(result.reason, "source_verified_override")

    def test_document_page_resolver_rejects_wrong_override_identity(self):
        for program, filename in [("DSBA", "other_page_028.png"), ("AIT", "dsba_page_028.png")]:
            with self.subTest(program=program, filename=filename):
                result = resolve_document_page(program, 28, filename, ["header", "content"])
                self.assertIsNone(result.document_page)
                self.assertEqual(result.reason, "unresolved")

    def test_document_page_resolver_fails_closed_on_unique_override_conflict(self):
        result = resolve_document_page(
            "DSBA", 28, "dsba_page_028.png", ["24", "header"]
        )
        self.assertIsNone(result.document_page)
        self.assertEqual(result.reason, "conflict")

    def test_document_page_resolver_uses_matching_override_for_ambiguous_ocr(self):
        result = resolve_document_page(
            "DSBA", 28, "dsba_page_028.png", ["23", "header", "content", "footer", "24"]
        )
        self.assertEqual(result.document_page, 23)
        self.assertEqual(result.reason, "source_verified_override")

    def test_document_page_resolver_keeps_unknown_and_legacy_pages_unresolved(self):
        for source_page, filename in [(999, "dsba_page_999.png"), (None, "legacy_ocr.json")]:
            with self.subTest(source_page=source_page, filename=filename):
                result = resolve_document_page("DSBA", source_page, filename, ["header"])
                self.assertIsNone(result.document_page)
                self.assertEqual(result.reason, "unresolved")


if __name__ == "__main__":
    unittest.main()
