"""H27: category-aware curriculum sum_credits (explicit single term only).

Bounded slice per H27-A decision A:
- category + sum_credits + exactly one year + exactly one semester
  -> filtered total over the canonical category-filtered course set.
- category + sum_credits without a concrete single term
  -> fail closed (partial -> insufficient_evidence), 0 model calls.
- all-category sums byte-identical.
"""

import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

from rag.qa import _classify_structured_parse_completeness, ask
from rag.query_spec import parse_query_spec
from rag.resolution import resolve_query_spec
from rag.structured.loader import load_json_to_sqlite

DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)

ELECTIVE_TERM_QUESTION = "IT ปี 4 เทอม 1 วิชาเลือก รวมทั้งหมดกี่หน่วยกิต"
GENED_TERM_QUESTION = "IT ปี 1 เทอม 1 หมวดวิชาศึกษาทั่วไป รวมทั้งหมดกี่หน่วยกิต"
FLAGSHIP_PROGRAM_ONLY = "IT วิชาเลือก รวมทั้งหมดกี่หน่วยกิต"


def _no_model(prompt: str) -> str:
    raise AssertionError("category sum path must not call any model")


def _ask(question: str, db_path=DB_PATH) -> dict:
    return ask(
        db_path,
        question,
        structured_model_callable=_no_model,
        answer_model_callable=_no_model,
        intent_model_callable=_no_model,
    )


def _sum_claims_by_plan(result) -> dict:
    sums: dict = {}
    for claim in result.claims:
        if claim.operation == "sum_credits" and claim.status == "complete":
            sums[tuple(getattr(claim.effective_scope, "plans", ()))] = claim.value
    return sums


class H27GuardTests(unittest.TestCase):
    def _classify(self, question: str):
        spec = parse_query_spec(question)
        resolution = resolve_query_spec(spec, DB_PATH)
        return spec, _classify_structured_parse_completeness(spec, resolution)

    def test_explicit_single_term_is_complete(self):
        spec, result = self._classify(ELECTIVE_TERM_QUESTION)
        self.assertEqual(spec.category, "วิชาเลือก")
        self.assertIn("sum_credits", tuple(spec.operations))
        self.assertEqual(tuple(spec.years), (4,))
        self.assertEqual(tuple(spec.semesters), (1,))
        self.assertEqual(result.classification, "complete")

    def test_category_year_only_is_partial(self):
        _spec, result = self._classify("IT ปี 4 วิชาเลือก รวมทั้งหมดกี่หน่วยกิต")
        self.assertEqual(result.classification, "partial")
        self.assertEqual(tuple(result.missing_filters), ())

    def test_category_semester_only_is_partial(self):
        _spec, result = self._classify("IT เทอม 1 วิชาเลือก รวมทั้งหมดกี่หน่วยกิต")
        self.assertEqual(result.classification, "partial")
        self.assertEqual(tuple(result.missing_filters), ())

    def test_category_program_only_flagship_is_partial(self):
        spec, result = self._classify(FLAGSHIP_PROGRAM_ONLY)
        self.assertEqual(spec.category, "วิชาเลือก")
        self.assertIn("sum_credits", tuple(spec.operations))
        self.assertEqual(result.classification, "partial")
        self.assertEqual(tuple(result.missing_filters), ())

    def test_all_category_sum_stays_complete(self):
        _spec, result = self._classify("IT ปี 4 เทอม 1 รวมทั้งหมดกี่หน่วยกิต")
        self.assertEqual(result.classification, "complete")

    def test_category_list_stays_complete(self):
        _spec, result = self._classify("IT ปี 4 เทอม 1 มีวิชาเลือกอะไรบ้าง")
        self.assertEqual(result.classification, "complete")


class H27FilteredSumTests(unittest.TestCase):
    def test_elective_exact_term_filtered_totals(self):
        result = _ask(ELECTIVE_TERM_QUESTION)["result"]
        self.assertEqual(result.status, "answer")
        # Canonical filtered set: 53 pooled 3-credit electives per plan plus
        # one uniform 6-credit alternative group (minimum 1) on no_coop.
        self.assertEqual(_sum_claims_by_plan(result).get(("coop",)), 159)
        self.assertEqual(_sum_claims_by_plan(result).get(("no_coop",)), 165)
        for claim in result.claims:
            if claim.operation == "sum_credits" and claim.status == "complete":
                self.assertEqual(claim.effective_scope.category, "วิชาเลือก")
                self.assertTrue(claim.provenance)
        self.assertIn("วิชาเลือก", result.final_answer)

    def test_gened_exact_term_filtered_total(self):
        result = _ask(GENED_TERM_QUESTION)["result"]
        self.assertEqual(result.status, "answer")
        sums = _sum_claims_by_plan(result)
        self.assertTrue(sums)
        # Canonical filtered set: 2 + 1 + 3 credits on each plan.
        for value in sums.values():
            self.assertEqual(value, 6)
        for claim in result.claims:
            if claim.operation == "sum_credits" and claim.status == "complete":
                self.assertEqual(
                    claim.effective_scope.category, "หมวดวิชาศึกษาทั่วไป"
                )
                self.assertTrue(claim.provenance)
        self.assertIn("หมวดวิชาศึกษาทั่วไป", result.final_answer)

    def test_explicit_plan_isolation(self):
        coop = _ask(
            "IT แผนสหกิจ ปี 4 เทอม 1 วิชาเลือก รวมทั้งหมดกี่หน่วยกิต"
        )["result"]
        self.assertEqual(coop.status, "answer")
        self.assertEqual(_sum_claims_by_plan(coop), {("coop",): 159})
        no_coop = _ask(
            "IT แผนไม่สหกิจ ปี 4 เทอม 1 วิชาเลือก รวมทั้งหมดกี่หน่วยกิต"
        )["result"]
        self.assertEqual(no_coop.status, "answer")
        self.assertEqual(_sum_claims_by_plan(no_coop), {("no_coop",): 165})

    def test_program_only_year_only_semester_only_fail_closed(self):
        for question in (
            FLAGSHIP_PROGRAM_ONLY,
            "IT ปี 4 วิชาเลือก รวมทั้งหมดกี่หน่วยกิต",
            "IT เทอม 1 วิชาเลือก รวมทั้งหมดกี่หน่วยกิต",
        ):
            with self.subTest(question=question):
                result = _ask(question)["result"]
                self.assertEqual(result.status, "insufficient_evidence")

    def test_all_category_sum_unchanged(self):
        result = _ask("IT ปี 4 เทอม 1 รวมทั้งหมดกี่หน่วยกิต")["result"]
        self.assertEqual(result.status, "answer")
        # Regression pin: existing whole-term totals via the public seam
        # (aggregation dedups identical logical courses within a partition,
        # so coop reports 15 here while the raw term query sums 18).
        self.assertEqual(_sum_claims_by_plan(result).get(("coop",)), 15)
        self.assertEqual(_sum_claims_by_plan(result).get(("no_coop",)), 15)
        self.assertNotIn("วิชาเลือก", result.final_answer)

    def test_provenance_contains_only_filtered_evidence(self):
        filtered = _ask(ELECTIVE_TERM_QUESTION)["result"]
        whole = _ask("IT ปี 4 เทอม 1 รวมทั้งหมดกี่หน่วยกิต")["result"]
        whole_codes: set = set()
        for claim in whole.claims:
            if claim.operation != "sum_credits" or claim.status != "complete":
                continue
            for component in claim.evidence.components:
                if isinstance(component, Mapping) and component.get("course_code"):
                    whole_codes.add(component.get("course_code"))
        self.assertTrue(whole_codes)
        for claim in filtered.claims:
            if claim.operation != "sum_credits" or claim.status != "complete":
                continue
            codes: set = set()
            for component in claim.evidence.components:
                self.assertTrue(component.get("provenance"))
                if component.get("course_code"):
                    codes.add(component.get("course_code"))
                for member in component.get("alternative_courses", ()) or ():
                    if isinstance(member, Mapping) and member.get("course_code"):
                        codes.add(member.get("course_code"))
            self.assertTrue(codes)
            self.assertEqual(codes & whole_codes, set())


def _write_doc(directory: Path, name: str, document: dict) -> Path:
    path = directory / name
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def _prov(page: int) -> list[dict]:
    return [
        {
            "program": "IT",
            "source_filename": f"page-{page}.png",
            "source_page": page,
            "document_category": "plan",
        }
    ]


def _build_temp_db(directory: Path, courses: list[dict]) -> Path:
    document = {"program": "IT", "plan": "default", "courses": courses}
    input_path = _write_doc(directory, "curriculum.json", document)
    db_path = directory / "curriculum.db"
    if db_path.exists():
        db_path.unlink()
    load_json_to_sqlite(input_path, db_path)
    return db_path


class H27TempDbTests(unittest.TestCase):
    def test_missing_credit_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _build_temp_db(
                Path(tmp),
                [
                    {
                        "code": "T0000001",
                        "name_th": "Known elective",
                        "credits": "3(3-0-6)",
                        "type": "เลือก",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=1),
                    },
                    {
                        "code": "T0000002",
                        "name_th": "Unknown credit elective",
                        "credits": "",
                        "type": "เลือก",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=2),
                    },
                ],
            )
            result = _ask(
                "IT ปี 1 เทอม 1 วิชาเลือก รวมทั้งหมดกี่หน่วยกิต",
                db_path=db_path,
            )["result"]
            self.assertEqual(result.status, "insufficient_evidence")

    def test_uniform_alternative_counts_minimum_choices(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _build_temp_db(
                Path(tmp),
                [
                    {
                        "code": "T1000001 หรือ T1000002",
                        "name_th": "Choice A\nChoice B",
                        "credits": "3(3-0-6)",
                        "type": "เลือก",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=3),
                    },
                    {
                        "code": "T1000003",
                        "name_th": "Plain elective",
                        "credits": "2(2-0-4)",
                        "type": "เลือก",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=4),
                    },
                ],
            )
            result = _ask(
                "IT ปี 1 เทอม 1 วิชาเลือก รวมทั้งหมดกี่หน่วยกิต",
                db_path=db_path,
            )["result"]
            self.assertEqual(result.status, "answer")
            values = [
                claim.value
                for claim in result.claims
                if claim.operation == "sum_credits" and claim.status == "complete"
            ]
            self.assertEqual(values, [5])

    def test_mixed_alternative_member_credits_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _build_temp_db(
                Path(tmp),
                [
                    {
                        "code": "T2000001 หรือ T2000002",
                        "name_th": "Mixed A\nMixed B",
                        "credits": "3(3-0-6) หรือ 4(4-0-8)",
                        "type": "เลือก",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=5),
                    },
                ],
            )
            result = _ask(
                "IT ปี 1 เทอม 1 วิชาเลือก รวมทั้งหมดกี่หน่วยกิต",
                db_path=db_path,
            )["result"]
            self.assertEqual(result.status, "insufficient_evidence")

    def test_unknown_alternative_member_credit_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _build_temp_db(
                Path(tmp),
                [
                    {
                        "code": "T3000001 หรือ T3000002",
                        "name_th": "Known\nUnknown",
                        "credits": "3(3-0-6) หรือ ไม่ระบุ",
                        "type": "เลือก",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=6),
                    },
                ],
            )
            result = _ask(
                "IT ปี 1 เทอม 1 วิชาเลือก รวมทั้งหมดกี่หน่วยกิต",
                db_path=db_path,
            )["result"]
            self.assertEqual(result.status, "insufficient_evidence")

    def test_conflicting_duplicate_placements_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _build_temp_db(
                Path(tmp),
                [
                    {
                        "code": "T4000001",
                        "name_th": "Duplicated elective",
                        "credits": "3(3-0-6)",
                        "credits_override": "2",
                        "type": "เลือก",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=7),
                    },
                    {
                        "code": "T4000001",
                        "name_th": "Duplicated elective",
                        "credits": "3(3-0-6)",
                        "credits_override": "3",
                        "type": "เลือก",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=8),
                    },
                ],
            )
            result = _ask(
                "IT ปี 1 เทอม 1 วิชาเลือก รวมทั้งหมดกี่หน่วยกิต",
                db_path=db_path,
            )["result"]
            self.assertEqual(result.status, "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
