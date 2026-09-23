"""H23: deterministic per-course credit filter."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rag.answer import _credit_scope_text
from rag.evidence_planner import build_structural_scope
from rag.intent_compiler import IntentCompilerError, compile_intent_to_query_spec
from rag.intent_interpreter import parse_intent_payload
from rag.qa import _classify_structured_parse_completeness, ask
from rag.query_spec import parse_query_spec
from rag.resolution import resolve_query_spec
from rag.structured.loader import load_json_to_sqlite
from rag.structured.queries import scoped_course_set

DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


def _course_list_payload():
    return json.dumps(
        {
            "intent": "course_list_query",
            "proposed_program": None,
            "proposed_plans": [],
            "proposed_years": [],
            "proposed_semesters": [],
            "course_codes": [],
            "topic": None,
            "requested_facts": ["course_list"],
            "judgement_dimension": None,
            "unresolved": [],
        },
        ensure_ascii=False,
    )


class H23ParserSeamTests(unittest.TestCase):
    def test_numeric_credit_predicate_captured(self):
        spec = parse_query_spec("IT ปี 3 มีวิชาไหน 3 หน่วยกิตบ้าง")
        self.assertEqual(spec.credit_units, 3)

    def test_numeric_credit_predicate_keeps_list_only(self):
        spec = parse_query_spec("IT ปี 3 มีวิชาไหน 3 หน่วยกิตบ้าง")
        self.assertEqual(tuple(spec.operations), ("list",))

    def test_no_credit_wording_leaves_predicate_empty(self):
        spec = parse_query_spec("IT ปี 3 มีวิชาอะไรบ้าง")
        self.assertIsNone(spec.credit_units)
        self.assertEqual(tuple(spec.operations), ("list",))

    def test_explicit_total_wording_keeps_sum_credits(self):
        spec = parse_query_spec("IT ปี 3 รวมกี่หน่วยกิต")
        self.assertIsNone(spec.credit_units)
        self.assertIn("sum_credits", tuple(spec.operations))

    def test_non_integral_credit_value_stays_unparsed(self):
        spec = parse_query_spec("IT ปี 3 มีวิชาไหน 3.5 หน่วยกิตบ้าง")
        self.assertIsNone(spec.credit_units)

    def test_thai_digit_credit_value_parses_like_years(self):
        spec = parse_query_spec("IT ปี 3 มีวิชาไหน ๓ หน่วยกิตบ้าง")
        self.assertEqual(spec.credit_units, 3)

    def test_non_integral_keeps_partial_credit_residue(self):
        spec = parse_query_spec("IT ปี 3 มีวิชาไหน 3.5 หน่วยกิตบ้าง")
        resolution = resolve_query_spec(spec, DB_PATH)
        result = _classify_structured_parse_completeness(spec, resolution)
        self.assertEqual(result.classification, "partial")
        self.assertIn("credit_units", result.missing_filters)

    def test_captured_predicate_fully_specifies_credit_axis(self):
        spec = parse_query_spec("IT ปี 3 เทอม 1 มีวิชาไหน 3 หน่วยกิตบ้าง")
        resolution = resolve_query_spec(spec, DB_PATH)
        result = _classify_structured_parse_completeness(spec, resolution)
        self.assertEqual(result.classification, "complete")
        self.assertNotIn("credit_units", result.missing_filters)


class H23DeterministicFilterTests(unittest.TestCase):
    def test_flagship_term1_complete_zero_calls_filtered_provenance(self):
        question = "IT ปี 3 เทอม 1 มีวิชาไหน 3 หน่วยกิตบ้าง"
        calls: list[str] = []

        def fake_model(prompt: str) -> str:
            calls.append(prompt)
            return _course_list_payload()

        result = ask(DB_PATH, question, intent_model_callable=fake_model)["result"]
        self.assertEqual(calls, [])
        self.assertEqual(result.status, "answer")
        self.assertTrue(result.claims)
        for claim in result.claims:
            self.assertEqual(claim.operation, "list")
            self.assertEqual(claim.status, "complete")
            self.assertEqual(getattr(claim.effective_scope, "credit_units", None), 3)
            self.assertTrue(claim.provenance)
            value = claim.value
            entries = (value,) if isinstance(value, dict) else tuple(value or ())
            self.assertTrue(entries)
            for entry in entries:
                if entry.get("is_alternative"):
                    for member in entry.get("alternative_courses", ()):
                        raw = member.get("credits") or member.get("credits_raw") or ""
                        self.assertTrue(str(raw).strip().startswith("3"))
                    continue
                raw = entry.get("placement_credits", entry.get("credits", ""))
                # Effective 3 credits: placement_credits may be "3(...)" text.
                self.assertTrue(str(raw).strip().startswith("3"))
        # Predicate is part of the rendered scope text.
        self.assertIn("3 หน่วยกิต", result.final_answer)
        header = _credit_scope_text(result.claims[0].effective_scope, prefix="")
        self.assertIn("3 หน่วยกิต", header)

    def test_both_terms_fail_closed_on_unknown_credit(self):
        # IT year 3 spans a no_coop sem-2 NULL-credit placeholder (060164xx);
        # any unknown in scope must fail the whole pack closed with 0 calls.
        question = "IT ปี 3 มีวิชาไหน 3 หน่วยกิตบ้าง"
        calls: list[str] = []

        def fake_model(prompt: str) -> str:
            calls.append(prompt)
            return _course_list_payload()

        result = ask(DB_PATH, question, intent_model_callable=fake_model)["result"]
        self.assertEqual(calls, [])
        self.assertEqual(result.status, "insufficient_evidence")
        kinds = {claim.status for claim in result.claims}
        self.assertIn("insufficient_evidence", kinds)
        # The coop partition is still a complete filtered list with provenance.
        complete = [c for c in result.claims if c.status == "complete"]
        self.assertTrue(complete)
        for claim in complete:
            self.assertEqual(getattr(claim.effective_scope, "credit_units", None), 3)
            self.assertTrue(claim.provenance)

    def test_planner_maps_credit_predicate_to_scope(self):
        spec = parse_query_spec("IT ปี 3 เทอม 1 มีวิชาไหน 3 หน่วยกิตบ้าง")
        resolution = resolve_query_spec(spec, DB_PATH)
        scope = build_structural_scope(spec, resolution)
        self.assertEqual(scope.credit_units, 3)
        plain = parse_query_spec("IT ปี 3 เทอม 1 มีวิชาอะไรบ้าง")
        plain_resolution = resolve_query_spec(plain, DB_PATH)
        plain_scope = build_structural_scope(plain, plain_resolution)
        self.assertIsNone(plain_scope.credit_units)

    def test_explicit_total_keeps_sum_credits(self):
        result = ask(DB_PATH, "IT ปี 3 รวมกี่หน่วยกิต")["result"]
        self.assertEqual(result.status, "answer")
        self.assertTrue(
            any(claim.operation == "sum_credits" for claim in result.claims)
        )

    def test_non_credit_lists_unchanged(self):
        calls: list[str] = []

        def fake_model(prompt: str) -> str:
            calls.append(prompt)
            return _course_list_payload()

        result = ask(
            DB_PATH, "IT ปี 3 เทอม 1 มีวิชาอะไรบ้าง",
            intent_model_callable=fake_model,
        )["result"]
        self.assertEqual(calls, [])
        self.assertEqual(result.status, "answer")
        for claim in result.claims:
            if claim.operation == "list" and claim.status == "complete":
                self.assertIsNone(
                    getattr(claim.effective_scope, "credit_units", None)
                )

    def test_course_list_query_rejects_credit_shapes(self):
        spec = parse_query_spec("IT ปี 3 เทอม 1 มีวิชาไหน 3 หน่วยกิตบ้าง")
        interpretation = parse_intent_payload(_course_list_payload())
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(spec, interpretation)


class H23R1ListOnlyGuardTests(unittest.TestCase):
    def _assert_partial_zero_calls(self, question: str):
        spec = parse_query_spec(question)
        self.assertEqual(spec.credit_units, 3)
        resolution = resolve_query_spec(spec, DB_PATH)
        result = _classify_structured_parse_completeness(spec, resolution)
        self.assertEqual(result.classification, "partial")
        calls: list[str] = []

        def fake_model(prompt: str) -> str:
            calls.append(prompt)
            return _course_list_payload()

        response = ask(DB_PATH, question, intent_model_callable=fake_model)["result"]
        self.assertEqual(calls, [])
        self.assertEqual(response.status, "insufficient_evidence")

    def test_count_credit_combo_fails_closed(self):
        self._assert_partial_zero_calls("IT ปี 3 เทอม 1 3 หน่วยกิตกี่วิชา")

    def test_existence_credit_combo_fails_closed(self):
        self._assert_partial_zero_calls("IT ปี 3 เทอม 1 3 หน่วยกิตมีไหม")

    def test_sum_credit_combo_fails_closed(self):
        self._assert_partial_zero_calls(
            "IT ปี 3 เทอม 1 3 หน่วยกิตรวมกี่หน่วยกิต"
        )

    def test_placement_credit_combo_fails_closed(self):
        self._assert_partial_zero_calls(
            "IT ปี 3 เทอม 1 3 หน่วยกิตเรียนตอนไหน"
        )


class H32StructuredSeamCreditGuardTests(unittest.TestCase):
    """H32-B: credit-bearing partials must not enter SQL fallback seams.

    Frozen H23-R1 promises every non-list credit combo fails closed with
    0 calls. The None-callable path held, but a supplied
    structured_model_callable still entered the course-list / placement
    selector seams (1 structured call, credit filter bypassed). These
    tests pin the seam shut with a recording stub.
    """

    def _assert_structured_seam_closed(self, question: str):
        spec = parse_query_spec(question)
        self.assertEqual(spec.credit_units, 3)
        resolution = resolve_query_spec(spec, DB_PATH)
        result = _classify_structured_parse_completeness(spec, resolution)
        self.assertEqual(result.classification, "partial")
        calls: list[str] = []

        def counting_structured(prompt: str) -> str:
            calls.append(prompt)
            return "SELECT 1"

        def no_intent(prompt: str) -> str:
            raise AssertionError("intent interpreter must not run")

        def no_answer(prompt: str) -> str:
            raise AssertionError("answer synthesis must not run")

        response = ask(
            DB_PATH,
            question,
            structured_model_callable=counting_structured,
            answer_model_callable=no_answer,
            intent_model_callable=no_intent,
        )["result"]
        self.assertEqual(calls, [])
        self.assertEqual(response.status, "insufficient_evidence")

    def test_count_credit_combo_closed_with_structured_callable(self):
        self._assert_structured_seam_closed("IT ปี 3 เทอม 1 3 หน่วยกิตกี่วิชา")

    def test_existence_credit_combo_closed_with_structured_callable(self):
        self._assert_structured_seam_closed("IT ปี 3 เทอม 1 3 หน่วยกิตมีไหม")

    def test_placement_credit_combo_closed_with_structured_callable(self):
        self._assert_structured_seam_closed(
            "IT ปี 3 เทอม 1 3 หน่วยกิตเรียนตอนไหน"
        )

    def test_count_credit_combo_closed_without_any_callable(self):
        question = "IT ปี 3 เทอม 1 3 หน่วยกิตกี่วิชา"
        response = ask(DB_PATH, question)["result"]
        self.assertEqual(response.status, "insufficient_evidence")

    def test_supported_list_credit_filter_still_complete(self):
        # ("list",) + integral predicate stays on the deterministic 0-call
        # path; the seam guard must not close the supported H23 shape.
        question = "IT ปี 3 เทอม 1 มีวิชาไหน 3 หน่วยกิตบ้าง"

        def no_structured(prompt: str) -> str:
            raise AssertionError("supported list filter needs no SQL fallback")

        def no_intent(prompt: str) -> str:
            raise AssertionError("supported list filter needs no interpretation")

        result = ask(
            DB_PATH,
            question,
            structured_model_callable=no_structured,
            answer_model_callable=None,
            intent_model_callable=no_intent,
        )["result"]
        self.assertEqual(result.status, "answer")
        complete = [c for c in result.claims if c.status == "complete"]
        self.assertTrue(complete)
        for claim in complete:
            self.assertEqual(claim.operation, "list")
            self.assertEqual(
                getattr(claim.effective_scope, "credit_units", None), 3
            )
            self.assertTrue(claim.provenance)


def _write_doc(directory: Path, name: str, document: dict) -> Path:
    path = directory / name
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def _prov(program: str = "TST", page: int = 1) -> list[dict]:
    return [
        {
            "program": program,
            "source_filename": f"page-{page}.png",
            "source_page": page,
            "document_category": "plan",
        }
    ]


def _build_temp_db(
    directory: Path, courses: list[dict], *, name: str = "curriculum.db"
) -> Path:
    # Use a canonical plan_key ("default") so scoped_course_set accepts it.
    document = {"program": "TST", "plan": "default", "courses": courses}
    input_path = _write_doc(directory, f"{name}.json", document)
    db_path = directory / name
    if db_path.exists():
        db_path.unlink()
    load_json_to_sqlite(input_path, db_path)
    return db_path


class H23OverridePrecedenceTests(unittest.TestCase):
    def test_override_wins_over_course_credit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = _build_temp_db(
                tmp_path,
                [
                    {
                        "code": "T0000001",
                        "name_th": "Override course",
                        "credits": "2(2-0-4)",
                        "credits_override": "3",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=1),
                    },
                    {
                        "code": "T0000002",
                        "name_th": "Plain course",
                        "credits": "3(3-0-6)",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=2),
                    },
                ],
            )
            matched = scoped_course_set(
                db_path, "TST", "default", years=(1,), semesters=(1,),
                credit_units=3,
            )
            self.assertEqual(matched["status"], "ok")
            codes = {
                c.get("course_code") for c in matched["courses"]
                if not c.get("is_alternative")
            }
            self.assertEqual(codes, {"T0000001", "T0000002"})

            excluded = scoped_course_set(
                db_path, "TST", "default", years=(1,), semesters=(1,),
                credit_units=2,
            )
            # Override 3 wins: T0000001 (course 2, override 3) no longer
            # matches predicate 2, and T0000002 is 3; both excluded.
            self.assertEqual(excluded["status"], "no_data")
            codes2 = {
                c.get("course_code") for c in excluded["courses"]
                if not c.get("is_alternative")
            }
            # Override 3 wins: T0000001 no longer matches predicate 2.
            self.assertEqual(codes2, set())

    def test_zero_override_uses_explicit_none_checks(self):
        # `or`-truthiness would treat "0" as absent; explicit checks must keep it.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = _build_temp_db(
                tmp_path,
                [
                    {
                        "code": "T0000003",
                        "name_th": "Zero override",
                        "credits": "3(3-0-6)",
                        "credits_override": "0",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=3),
                    },
                ],
            )
            matched = scoped_course_set(
                db_path, "TST", "default", years=(1,), semesters=(1,),
                credit_units=0,
            )
            self.assertEqual(matched["status"], "ok")
            self.assertEqual(len(matched["courses"]), 1)
            missed = scoped_course_set(
                db_path, "TST", "default", years=(1,), semesters=(1,),
                credit_units=3,
            )
            self.assertEqual(missed["status"], "no_data")


class H23AlternativeGroupTests(unittest.TestCase):
    def test_matching_group_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = _build_temp_db(
                tmp_path,
                [
                    {
                        "code": "T1000001 หรือ T1000002",
                        "name_th": "Choice A\nChoice B",
                        "credits": "3(3-0-6)",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=4),
                    },
                ],
            )
            result = scoped_course_set(
                db_path, "TST", "default", years=(1,), semesters=(1,),
                credit_units=3,
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(result["courses"]), 1)
            self.assertTrue(result["courses"][0].get("is_alternative"))

            excluded = scoped_course_set(
                db_path, "TST", "default", years=(1,), semesters=(1,),
                credit_units=4,
            )
            self.assertEqual(excluded["status"], "no_data")

    def test_mixed_member_credits_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = _build_temp_db(
                tmp_path,
                [
                    {
                        "code": "T2000001 หรือ T2000002",
                        "name_th": "Mixed A\nMixed B",
                        "credits": "3(3-0-6) หรือ 4(4-0-8)",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=5),
                    },
                ],
            )
            result = scoped_course_set(
                db_path, "TST", "default", years=(1,), semesters=(1,),
                credit_units=3,
            )
            self.assertEqual(result["status"], "insufficient_evidence")

    def test_unknown_member_credit_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Explicitly empty credit forces NULL credit_units -> unknown.
            db_path2 = _build_temp_db(
                tmp_path,
                [
                    {
                        "code": "T3000003",
                        "name_th": "Null credit",
                        "credits": "",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=7),
                    },
                ],
                name="curriculum2.db",
            )
            result = scoped_course_set(
                db_path2, "TST", "default", years=(1,), semesters=(1,),
                credit_units=3,
            )
            self.assertEqual(result["status"], "insufficient_evidence")

    def test_null_credit_never_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = _build_temp_db(
                tmp_path,
                [
                    {
                        "code": "T4000001",
                        "name_th": "Null credit single",
                        "credits": "",
                        "year": 1,
                        "semester": 1,
                        "source_provenance": _prov(page=8),
                    },
                ],
            )
            result = scoped_course_set(
                db_path, "TST", "default", years=(1,), semesters=(1,),
                credit_units=3,
            )
            self.assertEqual(result["status"], "insufficient_evidence")
            # Without a predicate the row is still returned (no filtering).
            unfiltered = scoped_course_set(
                db_path, "TST", "default", years=(1,), semesters=(1,),
            )
            self.assertEqual(unfiltered["status"], "ok")


if __name__ == "__main__":
    unittest.main()
