import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.run import (
    _apply_edition_metadata,
    _corrected_paths_for_program,
)
from src.pipeline.tools.correction.corrector import correct_json_files
from src.pipeline.tools.merge.consolidator import (
    edition_filename_token,
    merge_consecutive_files,
)


class EditionArtifactIsolationTests(unittest.TestCase):
    editions = (
        {"catalog_key": "dsba-2565-coop", "academic_year": "2565"},
        {"catalog_key": "dsba-2568-coop", "academic_year": "2568"},
    )

    def _write_extracted_input(self, input_dir: Path) -> None:
        input_dir.mkdir(parents=True)
        records = {
            33: [{"code": "06000001", "credits": "3(3-0-6)"}],
            317: [],
        }
        for page, courses in records.items():
            path = input_dir / f"dsba_page_{page:03d}_extracted.json"
            path.write_text(
                json.dumps({"program": "DSBA", "plan": "coop", "courses": courses}),
                encoding="utf-8",
            )

    def test_two_editions_coexist_and_discover_by_canonical_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extracted = root / "extracted"
            merged_dir = root / "consolidated"
            corrected_dir = root / "final"
            self._write_extracted_input(extracted)

            merged_paths = []
            corrected_paths = []
            for metadata in self.editions:
                merge_consecutive_files(
                    input_dir=str(extracted),
                    output_dir=str(merged_dir),
                    plan_filter="coop",
                    prefix="dsba",
                    desc_pages="317",
                    catalog_key=metadata["catalog_key"],
                )
                token = edition_filename_token(metadata["catalog_key"])
                full_path = next(
                    (merged_dir / "dsba" / "coop" / "full").glob(f"*_{token}_full.json")
                )
                merged_path = _apply_edition_metadata([full_path], metadata)[0]
                merged_paths.append(merged_path)
                corrected_paths.extend(correct_json_files([merged_path], corrected_dir))

            a = _corrected_paths_for_program(
                corrected_dir, "dsba", catalog_key=self.editions[0]["catalog_key"]
            )
            b = _corrected_paths_for_program(
                corrected_dir, "dsba", catalog_key=self.editions[1]["catalog_key"]
            )
            self.assertEqual(a, [corrected_paths[0]])
            self.assertEqual(b, [corrected_paths[1]])
            self.assertNotEqual(merged_paths[0], merged_paths[1])
            self.assertNotEqual(corrected_paths[0], corrected_paths[1])
            self.assertTrue(all(path.is_file() for path in (*merged_paths, *corrected_paths)))
            for path, metadata in zip((*merged_paths, *corrected_paths), (*self.editions, *self.editions)):
                document = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(document["catalog"]["catalog_key"], metadata["catalog_key"])

    def test_legacy_name_and_safe_filesystem_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extracted = root / "extracted"
            self._write_extracted_input(extracted)
            output = root / "consolidated"
            merge_consecutive_files(
                input_dir=str(extracted),
                output_dir=str(output),
                plan_filter="coop",
                prefix="dsba",
                desc_pages="317",
            )
            legacy = output / "dsba" / "coop" / "full" / "merged_dsba_coop_full.json"
            self.assertTrue(legacy.is_file())

            unsafe_output = root / "unsafe"
            merge_consecutive_files(
                input_dir=str(extracted),
                output_dir=str(unsafe_output),
                plan_filter="coop",
                prefix="dsba",
                desc_pages="317",
                catalog_key="../../outside\\edition",
            )
            generated = list(unsafe_output.glob("**/full/*.json"))
            self.assertEqual(len(generated), 1)
            self.assertEqual(generated[0].parent.resolve(), (unsafe_output / "dsba" / "coop" / "full").resolve())
            self.assertIn(edition_filename_token("../../outside\\edition"), generated[0].stem)

    def test_explicit_discovery_uses_json_identity_not_year_in_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong = root / "dsba_2568_coop_corrected.json"
            wrong.write_text(
                json.dumps({"program": "DSBA", "plan": "coop", "courses": []}),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                _corrected_paths_for_program(
                    root, "dsba", catalog_key="dsba-2568-coop"
                )

    def test_duplicate_edition_merge_fails_instead_of_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extracted = root / "extracted"
            self._write_extracted_input(extracted)
            args = {
                "input_dir": str(extracted),
                "output_dir": str(root / "consolidated"),
                "plan_filter": "coop",
                "prefix": "dsba",
                "desc_pages": "317",
                "catalog_key": self.editions[0]["catalog_key"],
            }
            merge_consecutive_files(**args)
            with self.assertRaises(FileExistsError):
                merge_consecutive_files(**args)


if __name__ == "__main__":
    unittest.main()
