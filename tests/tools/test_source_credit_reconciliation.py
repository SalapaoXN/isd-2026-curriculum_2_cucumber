import inspect
import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.tools.extraction import tool as extraction_tool


class SourceCreditReconciliationTests(unittest.TestCase):
    def _course(
        self,
        *,
        program="IT",
        code="06016454",
        credits="",
        plan="coop",
        source_filename="it_page_354.png",
        source_page=354,
        document_category="description",
    ):
        return {
            "program": program,
            "plan": plan,
            "code": code,
            "name_th": "ชื่อวิชา",
            "credits": credits,
            "source_provenance": [
                {
                    "program": program,
                    "source_filename": source_filename,
                    "source_page": source_page,
                    "document_category": document_category,
                }
            ],
        }

    @staticmethod
    def _corrections():
        return {
            (
                "IT",
                "coop",
                "06016454",
                "it_page_354.png",
                354,
                "description",
            ): {
                "program": "IT",
                "plan": "coop",
                "course_code": "06016454",
                "credits": "3(3-0-6)",
                "source_verified": True,
                "source_filename": "it_page_354.png",
                "source_page": 354,
                "document_category": "description",
            }
        }

    def test_production_extraction_has_no_ground_truth_dependency(self):
        source = inspect.getsource(extraction_tool)
        self.assertNotIn("ground_truth", source.lower())
        self.assertNotIn("_load_authoritative_credit_lookup", source)

    def test_reviewed_artifact_has_one_exact_source_verified_record(self):
        corrections = extraction_tool._load_source_verified_credit_corrections()
        self.assertEqual(len(corrections), 1)
        record = next(iter(corrections.values()))
        self.assertEqual(record["course_code"], "06016454")
        self.assertEqual(record["credits"], "3(3-0-6)")
        self.assertIs(record["source_verified"], True)

    def test_it_credit_repairs_only_with_exact_source_identity(self):
        result = extraction_tool._reconcile_source_backed_credit(
            self._course(),
            self._corrections(),
            program="IT",
            plan="coop",
        )
        self.assertEqual(result["credits"], "3(3-0-6)")

    def test_repair_exposes_explicit_source_verified_provenance(self):
        result = extraction_tool._reconcile_source_backed_credit(
            self._course(),
            self._corrections(),
            program="IT",
            plan="coop",
        )
        self.assertTrue(result["credit_source_verified"])
        self.assertEqual(
            result["credit_source_provenance"],
            {
                "source_filename": "it_page_354.png",
                "source_page": 354,
                "document_category": "description",
            },
        )
        self.assertTrue(result["source_provenance"][0]["source_verified"])

    def test_wrong_source_identity_does_not_repair(self):
        for changes in (
            {"source_page": 355},
            {"source_filename": "it_page_355.png"},
            {"program": "BIT"},
            {"plan": "no_coop"},
            {"code": "06016455"},
        ):
            course = self._course(**changes)
            result = extraction_tool._reconcile_source_backed_credit(
                course,
                self._corrections(),
                program=course["program"],
                plan=course["plan"],
            )
            self.assertIs(result, course)
            self.assertEqual(result["credits"], "")

    def test_bit_06036135_does_not_receive_gt_value(self):
        course = self._course(
            program="BIT",
            code="06036135",
            source_filename="bit_page_252_ocr.json",
            source_page=252,
        )
        result = extraction_tool._reconcile_source_backed_credit(
            course,
            extraction_tool._load_source_verified_credit_corrections(),
            program="BIT",
            plan="coop",
        )
        self.assertIs(result, course)
        self.assertEqual(result["credits"], "")

    def test_unsupported_parenthetical_credit_remains_unresolved(self):
        course = self._course(code="C001", credits="(0-2-1)")
        result = extraction_tool._reconcile_source_backed_credit(
            course,
            self._corrections(),
            program="IT",
            plan="coop",
        )
        self.assertIs(result, course)
        self.assertEqual(result["credits"], "(0-2-1)")

    def test_existing_valid_credit_is_not_replaced(self):
        course = self._course(credits="4(3-0-6)")
        result = extraction_tool._reconcile_source_backed_credit(
            course,
            self._corrections(),
            program="IT",
            plan="coop",
        )
        self.assertIs(result, course)
        self.assertEqual(result["credits"], "4(3-0-6)")

    def test_matching_valid_credit_is_unchanged(self):
        course = self._course(credits="3(3-0-6)")
        result = extraction_tool._reconcile_source_backed_credit(
            course,
            self._corrections(),
            program="IT",
            plan="coop",
        )
        self.assertIs(result, course)

    def test_malformed_correction_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            path.write_text(
                json.dumps({"corrections": [{"program": "IT"}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                extraction_tool._load_source_verified_credit_corrections(path)


if __name__ == "__main__":
    unittest.main()
