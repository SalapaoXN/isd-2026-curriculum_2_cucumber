import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.tools.runtime_artifacts import (
    FINAL_DIR,
    _match_ground_truth,
    canonicalize_runtime_artifacts,
)


class RuntimeArtifactTests(unittest.TestCase):
    @staticmethod
    def _read_course(path: Path, course_code: str) -> dict:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        return next(course for course in document["courses"] if course["code"] == course_code)

    def test_canonicalization_preserves_raw_finals_descriptions_and_provenance(self):
        raw_path = FINAL_DIR / "bit2565_coop_final.json"
        raw_bytes = raw_path.read_bytes()
        raw_course = self._read_course(raw_path, "06036100")

        with tempfile.TemporaryDirectory() as directory:
            result = canonicalize_runtime_artifacts(output_dir=Path(directory) / "canonical")
            canonical_dir = Path(directory) / "canonical"
            current_course = self._read_course(
                canonical_dir / raw_path.name, "06036100"
            )
            shared_course = self._read_course(
                canonical_dir / "dsba2565_coop_final.json", "90641001"
            )

        self.assertEqual(raw_path.read_bytes(), raw_bytes)
        self.assertEqual(raw_course["name_en"], "INFORMATION TECHNOLOGV FUNDAMENTALS")
        self.assertEqual(current_course["name_en"], "INFORMATION TECHNOLOGY FUNDAMENTALS")
        self.assertEqual(current_course["desc_en"], raw_course["desc_en"])
        self.assertEqual(current_course["source_provenance"], raw_course["source_provenance"])
        self.assertEqual(shared_course["type"], "บังคับ")
        self.assertEqual((shared_course["year"], shared_course["semester"]), (1, 1))
        self.assertGreater(result.plan_gt_fields_applied, 0)
        self.assertGreater(result.shared_general_education_fields_applied, 0)

    def test_legacy_corrections_are_source_scoped_and_preserve_descriptions(self):
        raw_path = FINAL_DIR / "dsba2560_coop_final.json"
        raw_bytes = raw_path.read_bytes()
        raw_project = self._read_course(raw_path, "06026128")

        with tempfile.TemporaryDirectory() as directory:
            result = canonicalize_runtime_artifacts(output_dir=Path(directory) / "canonical")
            canonical_dir = Path(directory) / "canonical"
            corrected_project = self._read_course(
                canonical_dir / raw_path.name, "06026128"
            )
            corrected_shared = self._read_course(
                canonical_dir / "dsba2560_no_coop_final.json", "06026104"
            )

        self.assertEqual(raw_path.read_bytes(), raw_bytes)
        self.assertIn("FREE ELECTIVE COURSE", raw_project["name_en"])
        self.assertEqual(
            corrected_project["name_en"],
            "PROJECT IN DATA SCIENCE AND BUSINESS ANALYTICS 2",
        )
        self.assertEqual(corrected_project["credits"], "3(0-9-0)")
        self.assertEqual(corrected_project["desc_th"], raw_project["desc_th"])
        self.assertTrue(
            any(
                entry.get("source_filename") == "dsba2560_page_033.png"
                for entry in corrected_project["source_provenance"]
            )
        )
        self.assertEqual(corrected_shared["name_en"], "COMPUTER PROGRAMMING")
        self.assertEqual(result.legacy_corrections_applied, 9)

    def test_matching_does_not_use_fuzzy_course_codes(self):
        raw = [{"code": "0603610I", "name_en": "INFORMATION TECHNOLOGY", "year": 1, "semester": 1}]
        ground_truth = [{"code": "06036101", "name_en": "INFORMATION TECHNOLOGY", "year": 1, "semester": 1}]

        with self.assertRaisesRegex(ValueError, "no matching reviewed-final record"):
            _match_ground_truth(
                raw,
                ground_truth,
                identity=("bit-2565", "BIT", "coop"),
                require_all_ground_truth_codes=True,
            )

    def test_matching_rejects_ambiguous_duplicate_course_identities(self):
        raw = [{"code": "06036101", "name_en": "DAMAGED NAME", "year": 1, "semester": 1}]
        ground_truth = [
            {"code": "06036101", "name_en": "COURSE ONE", "year": 1, "semester": 1},
            {"code": "06036101", "name_en": "COURSE TWO", "year": 1, "semester": 1},
        ]

        with self.assertRaisesRegex(ValueError, "ambiguous Ground Truth course identity"):
            _match_ground_truth(
                raw,
                ground_truth,
                identity=("bit-2565", "BIT", "coop"),
                require_all_ground_truth_codes=False,
            )


if __name__ == "__main__":
    unittest.main()
