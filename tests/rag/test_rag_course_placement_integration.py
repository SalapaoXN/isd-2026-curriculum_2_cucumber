import unittest
from pathlib import Path
from unittest.mock import patch

from rag.answer import EMPTY_ANSWER, answer_question
from rag.grounded_answer import GroundedAnswerResult
from rag.qa import ask
from rag.resolution import CourseReferenceResolution, ResolutionOutcome
from rag.structured.qa import _three_course_sequence_structured_result
from rag.structured.queries import (
    course_placement,
    earliest_year_semester,
    earliest_year_semester_from_choices,
    get_semester_credits,
    parse_flexible_year_semester,
    placement_year_semester_choices,
    scoped_course_set,
    semester_credits_and_prerequisites,
)


DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)
SUBMISSION_DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "submission"
    / "curriculum.db"
)


def _claims(result, operation):
    return [
        claim
        for claim in result["result"].claims
        if claim.operation == operation
    ]


def _term_scope(claim):
    return (
        tuple(claim.effective_scope.years),
        tuple(claim.effective_scope.semesters),
    )


class CoursePlacementIntegrationTest(unittest.TestCase):
    def test_flexible_parser_handles_single_and_multiple_choices(self):
        self.assertEqual(parse_flexible_year_semester("4/1"), [(4, 1)])
        self.assertEqual(
            parse_flexible_year_semester("3/1, 3/2, 4/1"),
            [(3, 1), (3, 2), (4, 1)],
        )

    def test_earliest_comparison_uses_fixed_and_flexible_choices(self):
        self.assertEqual(
            earliest_year_semester(None, None, "3/1, 3/2, 4/1"),
            (3, 1),
        )
        self.assertEqual(
            placement_year_semester_choices(3, 1, "4/1"),
            [(3, 1)],
        )
        self.assertEqual(
            earliest_year_semester(3, 2, None),
            (3, 2),
        )

    def test_malformed_flexible_value_fails_safely(self):
        self.assertEqual(parse_flexible_year_semester("3/1, unknown"), [])
        self.assertEqual(parse_flexible_year_semester("6/1"), [])
        self.assertEqual(placement_year_semester_choices(None, None, None), [])

    def test_earliest_composition_handles_mixed_placements_without_guessing(self):
        mixed_placements = [
            {"year_semester_choices": [(4, 1)]},
            {"year_semester_choices": [(3, 2), (4, 1)]},
            {"year_semester_choices": []},
        ]
        self.assertEqual(
            [
                earliest_year_semester_from_choices(
                    placement["year_semester_choices"]
                )
                for placement in mixed_placements
            ],
            [(4, 1), (3, 2), None],
        )
        self.assertIsNone(
            earliest_year_semester_from_choices([(3, 1), ("4", 1)])
        )

    def test_scoped_course_set_preserves_empty_year_five_scope(self):
        result = scoped_course_set(
            DB_PATH,
            "BIT",
            "no_coop",
            years=(5,),
            semesters=(1,),
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(result["years"], (5,))
        self.assertEqual(result["semesters"], (1,))
        self.assertEqual(result["courses"], [])
        credit_result = get_semester_credits(DB_PATH, "BIT", "no_coop", 5, 1)
        self.assertEqual(credit_result["status"], "no_data")
        self.assertEqual(credit_result["year"], 5)
        self.assertEqual(credit_result["semester"], 1)

    def test_course_placement_preserves_raw_and_exposes_choices(self):
        result = course_placement(DB_PATH, "IT", "06016481", ["coop", "no_coop"])
        by_plan = {placement["plan_key"]: placement for placement in result["placements"]}

        self.assertEqual(by_plan["coop"]["flexible_year_semester_raw"], None)
        self.assertEqual(by_plan["coop"]["year_semester_choices"], [(3, 2)])
        self.assertEqual(
            by_plan["no_coop"]["flexible_year_semester_raw"],
            "3/1, 3/2, 4/1",
        )
        self.assertEqual(
            by_plan["no_coop"]["year_semester_choices"],
            [(3, 1), (3, 2), (4, 1)],
        )

    def test_composed_result_uses_choices_for_earliest_timing(self):
        result = ask(
            DB_PATH,
            "วิชา 06016481 ใน IT แบบสหกิจและแบบไม่สหกิจ อยู่ปีไหน เทอมไหน?",
        )
        claims = _claims(result, "placement")
        by_plan = {}
        for claim in claims:
            self.assertEqual(claim.status, "complete")
            self.assertTrue(claim.provenance)
            for row in claim.evidence:
                by_plan[row["plan_key"]] = row

        self.assertEqual(set(by_plan), {"coop", "no_coop"})
        self.assertEqual(
            by_plan["coop"]["year_semester_choices"], ((3, 2),)
        )
        self.assertEqual(
            min(by_plan["coop"]["year_semester_choices"]), (3, 2)
        )
        self.assertEqual(
            by_plan["no_coop"]["year_semester_choices"],
            ((3, 1), (3, 2), (4, 1)),
        )
        self.assertEqual(
            min(by_plan["no_coop"]["year_semester_choices"]), (3, 1)
        )
        self.assertEqual(
            {member["course_code"] for member in by_plan["coop"]["alternative_courses"]},
            {"06016481", "06016482"},
        )

    def test_it_placement_uses_deterministic_operation(self):
        result = ask(
            DB_PATH,
            "วิชา 06016481 ใน IT แบบสหกิจและแบบไม่สหกิจ อยู่ปีไหน เทอมไหน "
            "และรายละเอียดการจัดวางต่างกันอย่างไร?",
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        claims = _claims(result, "placement")
        by_plan = {}
        for claim in claims:
            if claim.status != "complete" or not claim.provenance:
                continue
            for row in claim.evidence:
                by_plan[row["plan_key"]] = row
        self.assertEqual(set(by_plan), {"coop", "no_coop"})
        self.assertEqual(
            by_plan["coop"]["year_semester_choices"], ((3, 2),)
        )
        self.assertEqual(
            by_plan["no_coop"]["year_semester_choices"],
            ((3, 1), (3, 2), (4, 1)),
        )
        for row in by_plan.values():
            self.assertEqual(
                {member["course_code"] for member in row["alternative_courses"]},
                {"06016481", "06016482"},
            )
            self.assertTrue(
                all(member["name_en"] for member in row["alternative_courses"])
            )

    def test_cross_plan_earliest_placement_uses_deterministic_operation(self):
        calls = []

        def forbidden_model(_prompt):
            calls.append(True)
            raise AssertionError("structured model must not be called")

        result = ask(
            DB_PATH,
            "ถ้าอยากลง DATA CENTER DESIGN (06016465) ให้เร็วที่สุดใน IT "
            "ควรเลือกแผนไหน และแต่ละแผนเปิดให้ลงช่วงใดบ้าง?",
            structured_model_callable=forbidden_model,
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        claims = _claims(result, "placement")
        earliest_by_plan = {}
        for claim in claims:
            self.assertEqual(claim.status, "complete")
            self.assertTrue(claim.provenance)
            for row in claim.evidence:
                earliest_by_plan[row["plan_key"]] = row
        self.assertEqual(set(earliest_by_plan), {"coop", "no_coop"})
        self.assertEqual(
            earliest_by_plan["coop"]["year_semester_choices"], ((4, 1),)
        )
        self.assertEqual(
            earliest_by_plan["no_coop"]["year_semester_choices"],
            ((3, 1), (3, 2), (4, 1)),
        )
        self.assertLess(
            min(earliest_by_plan["no_coop"]["year_semester_choices"]),
            min(earliest_by_plan["coop"]["year_semester_choices"]),
        )
        self.assertEqual(calls, [])

    def test_explicit_plan_flexible_placement_wording_is_deterministic(self):
        calls = []

        def forbidden_model(_prompt):
            calls.append(True)
            raise AssertionError("structured model must not be called")

        result = ask(
            DB_PATH,
            "แผน IT แบบไม่สหกิจเปิดให้ลง DATA CENTER DESIGN "
            "(06016465) ช่วงไหนได้บ้าง?",
            structured_model_callable=forbidden_model,
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        claims = _claims(result, "placement")
        self.assertTrue(claims)
        rows = [
            row
            for claim in claims
            if claim.status == "complete" and claim.provenance
            for row in claim.evidence
        ]
        self.assertTrue(rows)
        self.assertEqual({row["plan_key"] for row in rows}, {"no_coop"})
        self.assertEqual(
            {row["year_semester_choices"] for row in rows},
            {((3, 1), (3, 2), (4, 1))},
        )
        self.assertTrue(all(row["provenance"] for row in rows))
        self.assertEqual(calls, [])

    def test_two_course_cross_plan_comparison_is_deterministic(self):
        calls = []

        def forbidden_model(_prompt):
            calls.append(True)
            raise AssertionError("structured model must not be called")

        result = ask(
            DB_PATH,
            "ถ้าต้องวางแผนเรียน SERVER SIDE WEB DEVELOPMENT (06016418) "
            "และ DATA CENTER DESIGN (06016465) ให้เร็วที่สุดใน IT "
            "ควรเลือกแผนไหน และแต่ละวิชาเรียนได้ช่วงใด?",
            structured_model_callable=forbidden_model,
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        placement_by_course_plan = {}
        for claim in _claims(result, "placement"):
            self.assertEqual(claim.status, "complete")
            self.assertTrue(claim.provenance)
            for row in claim.evidence:
                for member in row.get("alternative_courses") or ():
                    placement_by_course_plan[
                        (member["course_code"], row["plan_key"])
                    ] = row
                if row.get("course_code"):
                    placement_by_course_plan[
                        (row["course_code"], row["plan_key"])
                    ] = row
        self.assertEqual(
            {
                key: row["year_semester_choices"]
                for key, row in placement_by_course_plan.items()
            },
            {
                ("06016418", "coop"): ((3, 1),),
                ("06016418", "no_coop"): ((3, 1),),
                ("06016465", "coop"): ((4, 1),),
                ("06016465", "no_coop"): ((3, 1), (3, 2), (4, 1)),
            },
        )
        earliest_by_course_plan = {}
        for claim in _claims(result, "earliest"):
            self.assertEqual(claim.status, "complete")
            for partition in claim.value.partitions:
                for placement in partition.placements:
                    earliest_by_course_plan[
                        (placement["course_code"], partition.partition["plans"][0])
                    ] = partition.value
        self.assertEqual(
            earliest_by_course_plan,
            {
                ("06016418", "coop"): (3, 1),
                ("06016418", "no_coop"): (3, 1),
                ("06016465", "coop"): (4, 1),
                ("06016465", "no_coop"): (3, 1),
            },
        )
        self.assertLess(
            earliest_by_course_plan[("06016465", "no_coop")],
            earliest_by_course_plan[("06016465", "coop")],
        )
        self.assertEqual(calls, [])

    def test_two_course_comparison_does_not_guess_ties_or_missing_timing(self):
        def placement_result(course_code, choices):
            placements = []
            for plan_key in ("coop", "no_coop"):
                placements.append(
                    {
                        "placement_id": len(placements) + 1,
                        "plan_key": plan_key,
                        "course_id": len(placements) + 1,
                        "course_code": course_code,
                        "year_semester_choices": choices,
                        "provenance": [{"provenance_id": len(placements) + 1}],
                    }
                )
            return {
                "status": "ok",
                "missing_plan_keys": [],
                "placements": placements,
            }

        with patch(
            "rag.structured.qa.course_placement",
            side_effect=lambda _db, _program, course_code, _plans: placement_result(
                course_code, [(3, 1)]
            ),
        ):
            tied = ask(
                DB_PATH,
                "สองวิชา 00000001 และ 00000002 ใน IT ให้เร็วที่สุด "
                "ควรเลือกแผนไหน?",
            )
        self.assertNotIn("earliest_plan", tied["result"])

        with patch(
            "rag.structured.qa.course_placement",
            side_effect=[
                placement_result("00000001", [(3, 1)]),
                placement_result("00000002", []),
            ],
        ):
            missing_timing = ask(
                DB_PATH,
                "สองวิชา 00000001 และ 00000002 ใน IT ให้เร็วที่สุด "
                "ควรเลือกแผนไหน?",
            )
        self.assertNotIn("earliest_plan", missing_timing["result"])

    def test_three_course_infrastructure_sequence_is_deterministic(self):
        calls = []

        def forbidden_model(_prompt):
            calls.append(True)
            raise AssertionError("structured model must not be called")

        result = ask(
            DB_PATH,
            "ถ้าจะวางแผนเรียนสาย infrastructure ใน IT แบบไม่สหกิจ "
            "ควรเรียง INTRODUCTION TO NETWORK SYSTEMS (06016413), "
            "INFRASTRUCTURE SYSTEMS AND SERVICES (06016420) และ "
            "INFORMATION TECHNOLOGY INFRASTRUCTURE SECURITY (06016421) "
            "ตามปี/เทอมอย่างไร และแต่ละวิชาต้องผ่านวิชาอะไรมาก่อน?",
            structured_model_callable=forbidden_model,
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        prereq_by_term = {}
        for claim in _claims(result, "prerequisite"):
            prereq_by_term[_term_scope(claim)] = claim
        self.assertEqual(
            sorted(prereq_by_term),
            [((2,), (1,)), ((2,), (2,)), ((3,), (1,))],
        )
        self.assertEqual(prereq_by_term[((2,), (1,))].status, "valid_empty")
        self.assertEqual(
            [
                item["prerequisite_code"]
                for item in prereq_by_term[((2,), (2,))].evidence
            ],
            ["06016413"],
        )
        self.assertEqual(
            [
                item["prerequisite_code"]
                for item in prereq_by_term[((3,), (1,))].evidence
            ],
            ["06016413"],
        )
        self.assertTrue(
            all(
                claim.provenance
                for term, claim in prereq_by_term.items()
                if claim.status == "complete"
            )
        )
        describe_codes = [
            row["course_code"]
            for claim in _claims(result, "describe")
            if claim.status == "complete" and claim.provenance
            for row in claim.evidence
        ]
        self.assertEqual(
            set(describe_codes), {"06016413", "06016420", "06016421"}
        )
        self.assertEqual(calls, [])

    def test_three_course_sequence_sorts_ties_and_leaves_missing_timing_unknown(self):
        # Sequencing is a deterministic composition over already-grounded
        # placement facts, so exercise that seam directly with in-memory
        # placement results instead of fabricating provenance via qa.ask().
        def placement_result(course_code, choices):
            return {
                "status": "ok",
                "course_code": course_code,
                "missing_plan_keys": [],
                "placements": [
                    {
                        "placement_id": int(course_code[-1]),
                        "plan_key": "no_coop",
                        "course_id": int(course_code[-1]),
                        "course_code": course_code,
                        "year_semester_choices": choices,
                        "provenance": [{"provenance_id": int(course_code[-1])}],
                    }
                ],
            }

        choices = {
            "00000001": [(3, 1)],
            "00000002": [(3, 1)],
            "00000003": [],
        }
        course_codes = ["00000001", "00000002", "00000003"]
        sequence = _three_course_sequence_structured_result(
            DB_PATH,
            [placement_result(code, choices[code]) for code in course_codes],
            "IT",
            course_codes,
            ["no_coop"],
        )

        self.assertEqual(sequence["operation"], "course_sequence")
        self.assertEqual(sequence["status"], "ok")
        courses = sequence["derived_facts"]["sequence_by_plan"][0]["courses"]
        self.assertEqual(
            [item["course_code"] for item in courses],
            ["00000001", "00000002", "00000003"],
        )
        self.assertEqual(
            [item["earliest_year_semester"] for item in courses],
            [(3, 1), (3, 1), None],
        )
        self.assertEqual(
            [item["sequence_order"] for item in courses],
            [1, 2, 3],
        )
        self.assertTrue(
            all(item["prerequisites"] == [] for item in courses)
        )

    def test_course_name_from_placement_reaches_grounded_answer(self):
        question = (
            "วิชา 06016414 ของ IT แบบไม่สหกิจชื่อภาษาอังกฤษว่าอะไร "
            "และมีหน่วยกิตเท่าไร?"
        )
        result = ask(DB_PATH, question)
        self.assertIsInstance(result["result"], GroundedAnswerResult)
        credit_claims = [
            claim
            for claim in _claims(result, "sum_credits")
            if claim.status == "complete" and claim.provenance
        ]
        self.assertTrue(credit_claims)
        components = [
            component
            for claim in credit_claims
            for component in (getattr(claim.evidence, "components", None) or ())
            if component.get("course_code") == "06016414"
        ]
        self.assertTrue(components)
        self.assertTrue(all(component["name_en"] for component in components))
        self.assertEqual(
            {component["name_en"] for component in components},
            {"NOSQL DATABASE SYSTEMS"},
        )
        self.assertEqual(
            {component["counted_credit_units"] for component in components},
            {3},
        )

        prompts = []
        answers = iter([EMPTY_ANSWER, "คำตอบจากข้อมูลวิชา"])

        def answer_model(prompt):
            prompts.append(prompt)
            return next(answers)

        rows = [
            {"name_en": component["name_en"], "credit_units": 3}
            for component in components
        ]
        answer = answer_question(
            question,
            "structured",
            structured_result={
                "operation": "course_facts",
                "status": "ok",
                "columns": ["name_en", "credit_units"],
                "rows": [(row["name_en"], row["credit_units"]) for row in rows],
            },
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, "คำตอบจากข้อมูลวิชา")
        self.assertEqual(len(prompts), 2)
        self.assertIn("name_en", prompts[0])
        self.assertIn(rows[0]["name_en"], prompts[0])

    def test_bit_placement_preserves_both_independent_plan_rows(self):
        result = ask(
            DB_PATH,
            "วิชา 06036103 ใน BIT แบบสหกิจและแบบไม่สหกิจ อยู่ปีไหนและเทอมไหน?",
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        rows = [
            row
            for claim in _claims(result, "placement")
            if claim.status == "complete" and claim.provenance
            for row in claim.evidence
        ]
        self.assertEqual(
            {
                row["plan_key"]: (
                    row["course_id"],
                    row["year_number"],
                    row["semester_number"],
                )
                for row in rows
            },
            {"coop": (68, 2, 1), "no_coop": (129, 2, 1)},
        )

    def test_non_placement_structured_question_keeps_nl_to_sql_fallback(self):
        calls = []

        def fake_model(_prompt):
            calls.append(True)
            return "SELECT 1"

        result = ask(
            DB_PATH,
            "รวมกี่หน่วยกิต?",
            structured_model_callable=fake_model,
        )

        self.assertIsNone(result["route"])
        self.assertIsInstance(result["result"], GroundedAnswerResult)
        self.assertEqual(result["result"].status, "insufficient_evidence")
        self.assertEqual(calls, [])

    def test_non_plan_sensitive_course_credit_question_is_deterministic(self):
        def fail_model(_prompt):
            self.fail("explicit course credit facts must not call the model")

        result = ask(
            DB_PATH,
            "IT วิชา 06016465 มีกี่หน่วยกิต?",
            structured_model_callable=fail_model,
        )

        self.assertIsNone(result["route"])
        self.assertIsInstance(result["result"], GroundedAnswerResult)
        claims = _claims(result, "sum_credits")
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].status, "complete")
        self.assertEqual(claims[0].value, 3)
        self.assertTrue(claims[0].provenance)
        self.assertEqual(
            tuple(claims[0].effective_scope.plans), ("coop", "no_coop")
        )

    def test_exact_course_name_credit_facts_are_deterministic_and_grounded(self):
        # Program scope is required for name+credit operations (fail-closed
        # program guard); bare course-code wording clarifies instead.
        questions = (
            (
                "IT วิชา 06016401 ชื่อภาษาอังกฤษว่าอะไรและมีหน่วยกิตเท่าไร?",
                "MATHEMATICS FOR INFORMATION TECHNOLOGY",
            ),
            (
                "NOSQL DATABASE SYSTEMS (06016414) มีหน่วยกิตเท่าไรในหลักสูตร IT?",
                "NOSQL DATABASE SYSTEMS",
            ),
            (
                "IT วิชา 06016420 ชื่ออะไรและมีกี่หน่วยกิต?",
                "INFRASTRUCTURE SYSTEMS AND SERVICES",
            ),
        )

        for question, expected_name in questions:
            with self.subTest(question=question):
                def fail_model(_prompt):
                    self.fail("course facts must not call the structured model")

                result = ask(
                    SUBMISSION_DB_PATH,
                    question,
                    structured_model_callable=fail_model,
                )
                self.assertIsNone(result["route"])
                self.assertIsInstance(result["result"], GroundedAnswerResult)
                names = set()
                credits = set()
                provenances = []
                for claim in result["result"].claims:
                    if claim.operation == "identity":
                        for item in claim.value:
                            names.add(item["name_en"])
                            provenances.append(item["provenance"])
                    if claim.operation == "sum_credits":
                        for component in (
                            getattr(claim.evidence, "components", None) or ()
                        ):
                            names.add(component["name_en"])
                            credits.add(component["counted_credit_units"])
                    if claim.provenance:
                        provenances.append(claim.provenance)
                self.assertEqual(names, {expected_name})
                self.assertEqual(credits, {3})
                self.assertTrue(provenances)
                self.assertTrue(
                    all(provenance for provenance in provenances)
                )

    def test_exact_unknown_course_facts_are_deterministic_no_data(self):
        def fail_model(_prompt):
            self.fail("unknown exact course facts must not call the structured model")

        result = ask(
            DB_PATH,
            "วิชา 99999999 ชื่ออะไรและมีกี่หน่วยกิต?",
            structured_model_callable=fail_model,
        )

        self.assertIsNone(result["route"])
        self.assertEqual(result["result"]["status"], "no_data")
        self.assertEqual(result["result"]["action"], "no_data")
        self.assertEqual(result["result"]["blocking_ambiguity"], ())
        self.assertEqual(result["result"]["course_references"][0]["candidates"], [])

    def test_prerequisite_only_is_deterministic_but_credits_only_is_unchanged(self):
        calls = []

        def fake_model(_prompt):
            calls.append(True)
            return "SELECT 1"

        prerequisite_result = ask(
            DB_PATH,
            "IT แบบไม่สหกิจ วิชา 06016420 ต้องเรียนก่อนวิชาอะไร?",
            structured_model_callable=fake_model,
        )
        credits_result = ask(
            DB_PATH,
            "IT แบบไม่สหกิจ ในปี 2 เทอม 2 ลงทะเบียนรวมกี่หน่วยกิต",
            structured_model_callable=fake_model,
        )

        # "ต้องเรียนก่อนวิชาอะไร" asks for successor/dependent courses.
        # Nothing requires 06016420, so the successor relation is empty:
        # placement context is grounded, but no prerequisite claim appears.
        self.assertIsInstance(
            prerequisite_result["result"], GroundedAnswerResult
        )
        placements = _claims(prerequisite_result, "placement")
        self.assertEqual(len(placements), 1)
        self.assertEqual(placements[0].status, "complete")
        self.assertTrue(placements[0].provenance)
        self.assertEqual(
            placements[0].evidence[0]["course_code"], "06016420"
        )
        self.assertEqual(
            _claims(prerequisite_result, "prerequisite"), []
        )
        sums = _claims(credits_result, "sum_credits")
        self.assertEqual(len(sums), 1)
        self.assertEqual(sums[0].status, "complete")
        self.assertEqual(sums[0].value, 30)
        self.assertTrue(sums[0].provenance)
        self.assertEqual(len(calls), 0)

    def test_course_without_prerequisite_returns_no_data_without_model(self):
        calls = []

        def fake_model(_prompt):
            calls.append(True)
            raise AssertionError("structured model must not be called")

        result = ask(
            DB_PATH,
            "IT แบบไม่สหกิจ วิชา 06016465 ต้องเรียนก่อนวิชาอะไร?",
            structured_model_callable=fake_model,
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        placements = _claims(result, "placement")
        self.assertEqual(len(placements), 1)
        self.assertEqual(placements[0].status, "complete")
        self.assertTrue(placements[0].provenance)
        self.assertEqual(
            placements[0].evidence[0]["course_code"], "06016465"
        )
        self.assertEqual(
            placements[0].evidence[0]["year_semester_choices"],
            ((3, 1), (3, 2), (4, 1)),
        )
        # 06016465 has no prerequisites and no successors: the typed
        # result carries placement context with no prerequisite claims.
        self.assertEqual(_claims(result, "prerequisite"), [])
        self.assertEqual(calls, [])

    def test_semester_course_list_is_deterministic_and_preserves_order(self):
        calls = []

        def forbidden_model(_prompt):
            calls.append(True)
            raise AssertionError("structured model must not be called")

        result = ask(
            DB_PATH,
            "IT แบบไม่สหกิจ ปี 1 เทอม 1 ต้องเรียนวิชาอะไรบ้าง?",
            structured_model_callable=forbidden_model,
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        rows = [
            row
            for claim in _claims(result, "list")
            if claim.status == "complete" and claim.provenance
            for row in claim.value
        ]
        self.assertEqual(
            [row["course_code"] for row in rows],
            [
                "06016401",
                "06016402",
                "06016411",
                "06066303",
                "90641001",
                "90641003",
                "90644007",
            ],
        )
        self.assertTrue(
            all(row["plan_key"] == "no_coop" for row in rows)
        )
        self.assertTrue(all(row["provenance"] for row in rows))
        self.assertEqual(calls, [])

    def test_semester_course_list_without_plan_returns_both_plans(self):
        calls = []

        def forbidden_model(_prompt):
            calls.append(True)
            raise AssertionError("structured model must not be called")

        result = ask(
            DB_PATH,
            "IT ปี 1 เทอม 1 ต้องเรียนวิชาอะไรบ้าง?",
            structured_model_callable=forbidden_model,
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        by_plan = {}
        for claim in _claims(result, "list"):
            self.assertEqual(claim.status, "complete")
            self.assertTrue(claim.provenance)
            for row in claim.value:
                by_plan.setdefault(row["plan_key"], []).extend(
                    [row["course_code"]]
                )
        self.assertEqual(set(by_plan), {"coop", "no_coop"})
        expected = [
            "06016401",
            "06016402",
            "06016411",
            "06066303",
            "90641001",
            "90641003",
            "90644007",
        ]
        self.assertEqual(by_plan["coop"], expected)
        self.assertEqual(by_plan["no_coop"], expected)
        self.assertEqual(calls, [])

    def test_semester_course_list_preserves_alternative_group(self):
        calls = []

        def forbidden_model(_prompt):
            calls.append(True)
            raise AssertionError("structured model must not be called")

        result = ask(
            DB_PATH,
            "IT แบบสหกิจ ปี 3 เทอม 2 ต้องเรียนวิชาอะไรบ้าง?",
            structured_model_callable=forbidden_model,
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        groups = [
            row
            for claim in _claims(result, "list")
            if claim.status == "complete" and claim.provenance
            for row in claim.value
            if row.get("is_alternative")
        ]
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            [member["course_code"] for member in groups[0]["alternative_courses"]],
            ["06016481", "06016482"],
        )
        self.assertTrue(groups[0]["provenance"])
        self.assertEqual(calls, [])

    def test_gold_alternative_group_question_is_deterministic(self):
        calls = []

        def forbidden_model(_prompt):
            calls.append(True)
            raise AssertionError("structured model must not be called")

        result = ask(
            DB_PATH,
            "ใน IT แบบสหกิจ กลุ่ม 06016481 กับ 06016482 ต้องเลือกกี่วิชา "
            "และเรียนช่วงไหน?",
            structured_model_callable=forbidden_model,
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        groups = {}
        for claim in _claims(result, "count"):
            self.assertEqual(claim.status, "complete")
            self.assertTrue(claim.provenance)
            self.assertEqual(claim.value, 1)
            for row in claim.evidence.courses:
                groups[row["plan_key"]] = row
        self.assertEqual(set(groups), {"coop"})
        self.assertEqual(groups["coop"]["year_semester_choices"], ((3, 2),))
        self.assertEqual(groups["coop"]["minimum_choices"], 1)
        self.assertEqual(groups["coop"]["maximum_choices"], 1)
        self.assertEqual(
            {member["course_code"] for member in groups["coop"]["alternative_courses"]},
            {"06016481", "06016482"},
        )
        self.assertTrue(groups["coop"]["provenance"])
        self.assertEqual(calls, [])

    def test_gold_cross_plan_alternative_group_question_is_deterministic(self):
        calls = []

        def forbidden_model(_prompt):
            calls.append(True)
            raise AssertionError("structured model must not be called")

        result = ask(
            DB_PATH,
            "ถ้าต้องเลือกระหว่างกลุ่มวิชา 06016481 กับ 06016482 ใน IT "
            "แผนสหกิจกับไม่สหกิจต่างกันอย่างไร ทั้งจำนวนวิชาที่เลือก "
            "และช่วงเรียน?",
            structured_model_callable=forbidden_model,
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        groups = {}
        for claim in _claims(result, "count"):
            self.assertEqual(claim.status, "complete")
            self.assertTrue(claim.provenance)
            self.assertEqual(claim.value, 1)
            for row in claim.evidence.courses:
                groups[(row["plan_key"], row["placement_id"])] = row
        self.assertEqual(
            {plan for plan, _ in groups}, {"coop", "no_coop"}
        )
        for row in groups.values():
            self.assertEqual(row["minimum_choices"], 1)
            self.assertEqual(row["maximum_choices"], 1)
            self.assertEqual(
                {member["course_code"] for member in row["alternative_courses"]},
                {"06016481", "06016482"},
            )
            self.assertTrue(row["provenance"])
        by_plan = {}
        for (plan, _), row in groups.items():
            by_plan.setdefault(plan, row["year_semester_choices"])
        self.assertEqual(by_plan["coop"], ((3, 2),))
        self.assertEqual(
            by_plan["no_coop"], ((3, 1), (3, 2), (4, 1))
        )
        self.assertEqual(calls, [])

    def test_semester_credits_operation_resolves_plan_and_components(self):
        result = get_semester_credits(DB_PATH, "IT", "coop", 2, 2)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total_credits"], 30)
        self.assertTrue(result["components"])
        self.assertTrue(
            all(component["plan_key"] == "coop" for component in result["components"])
        )
        self.assertTrue(
            all(component["year"] == 2 for component in result["components"])
        )
        self.assertTrue(
            all(component["semester"] == 2 for component in result["components"])
        )

        response = ask(DB_PATH, "IT แบบสหกิจ ปี 2 เทอม 2 รวมกี่หน่วยกิต")
        self.assertIsInstance(response["result"], GroundedAnswerResult)
        sums = _claims(response, "sum_credits")
        self.assertEqual(len(sums), 1)
        self.assertEqual(sums[0].status, "complete")
        self.assertEqual(sums[0].value, 30)
        self.assertTrue(sums[0].provenance)

    def test_semester_credits_alternative_group_counts_once(self):
        result = get_semester_credits(DB_PATH, "IT", "coop", 3, 2)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total_credits"], 6)
        alternatives = [
            component
            for component in result["components"]
            if component["alternative_group_id"] is not None
        ]
        self.assertEqual(len(alternatives), 1)
        self.assertEqual(alternatives[0]["counted_credit_units"], 6)

        response = ask(DB_PATH, "IT แบบสหกิจ ปี 3 เทอม 2 รวมกี่หน่วยกิต")
        self.assertIsInstance(response["result"], GroundedAnswerResult)
        sums = _claims(response, "sum_credits")
        self.assertEqual(len(sums), 1)
        self.assertEqual(sums[0].status, "complete")
        self.assertEqual(sums[0].value, 6)
        self.assertTrue(sums[0].provenance)
        group_components = [
            component
            for component in sums[0].evidence.components
            if component.get("alternative_group_id") is not None
        ]
        self.assertEqual(len(group_components), 1)
        self.assertEqual(group_components[0]["counted_credit_units"], 6)

    def test_semester_credits_missing_term_is_no_data(self):
        result = get_semester_credits(DB_PATH, "IT", "default", 1, 1)

        self.assertEqual(result["status"], "no_data")
        self.assertIsNone(result["total_credits"])
        self.assertEqual(result["components"], [])

    def test_semester_credits_and_prerequisite_use_deterministic_operation(self):
        result = ask(
            DB_PATH,
            "IT แบบไม่สหกิจ ในปี 2 เทอม 2 ลงทะเบียนรวมกี่หน่วยกิต "
            "และวิชา 06016420 ต้องผ่านวิชาอะไรมาก่อน?"
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        sums = _claims(result, "sum_credits")
        self.assertEqual(len(sums), 1)
        self.assertEqual(sums[0].status, "complete")
        # "รวมกี่หน่วยกิต" is an explicit semester total: the frozen planner
        # rule (_credit_request_targets) drops exact-course targets, so the
        # term aggregate (10 x 3) is authoritative, not the mentioned course.
        self.assertEqual(sums[0].value, 30)
        self.assertTrue(sums[0].provenance)
        component_codes = [
            component["course_code"]
            for component in sums[0].evidence.components
        ]
        self.assertIn("06016420", component_codes)
        self.assertEqual(
            sums[0].value,
            sum(
                component["counted_credit_units"]
                for component in sums[0].evidence.components
            ),
        )
        prereqs = _claims(result, "prerequisite")
        self.assertEqual(len(prereqs), 1)
        self.assertEqual(prereqs[0].status, "complete")
        self.assertTrue(prereqs[0].provenance)
        rows = list(prereqs[0].evidence)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["course_id"], 739)
        self.assertEqual(rows[0]["prerequisite_course_id"], 728)
        self.assertEqual(rows[0]["prerequisite_code"], "06016413")
        self.assertEqual(rows[0]["requirement_type"], "required")
        self.assertEqual(rows[0]["raw_text"], "06016413")
        self.assertTrue(rows[0]["provenance"])

    def test_hard_planning_question_uses_mixed_deterministic_operation(self):
        calls = []

        def forbidden_model(_prompt):
            calls.append(True)
            raise AssertionError("structured model must not be called")

        result = ask(
            DB_PATH,
            "ถ้าจะลง INFRASTRUCTURE SYSTEMS AND SERVICES (06016420) ใน IT "
            "แบบไม่สหกิจปี 2 เทอม 2 ต้องเตรียมผ่านวิชาอะไรในเทอมก่อนหน้า "
            "และเทอมนี้มีหน่วยกิตรวมเท่าไร?",
            structured_model_callable=forbidden_model,
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        sums = _claims(result, "sum_credits")
        self.assertEqual(len(sums), 1)
        self.assertEqual(sums[0].status, "complete")
        # Exact-course scope: the mentioned course carries its own credits.
        self.assertEqual(sums[0].value, 3)
        self.assertTrue(sums[0].provenance)
        prereqs = _claims(result, "prerequisite")
        self.assertEqual(len(prereqs), 1)
        self.assertEqual(prereqs[0].status, "complete")
        rows = list(prereqs[0].evidence)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prerequisite_code"], "06016413")
        self.assertEqual(rows[0]["requirement_type"], "required")
        self.assertEqual(rows[0]["raw_text"], "06016413")
        self.assertTrue(rows[0]["provenance"])
        self.assertEqual(calls, [])

    def test_placement_semantic_hybrid_keeps_both_evidence_paths(self):
        question = (
            "วิชา 06016481 ใน IT แบบสหกิจและแบบไม่สหกิจ อยู่ปีไหน เทอมไหน "
            "และเนื้อหาเกี่ยวข้องกับสถานประกอบการอย่างไร?"
        )
        result = ask(DB_PATH, question)

        self.assertIsNone(result["route"])
        self.assertIsInstance(result["result"], GroundedAnswerResult)
        placements = _claims(result, "placement")
        self.assertTrue(placements)
        self.assertTrue(
            all(
                claim.status == "complete" and claim.provenance
                for claim in placements
            )
        )
        self.assertEqual(
            {
                row["plan_key"]
                for claim in placements
                for row in claim.evidence
            },
            {"coop", "no_coop"},
        )
        describes = [
            claim
            for claim in _claims(result, "describe")
            if claim.status == "complete" and claim.provenance
        ]
        self.assertTrue(describes)
        chunk_courses = {
            row["course_code"]
            for claim in describes
            for row in claim.evidence
        }
        self.assertIn("06016481", chunk_courses)

    def test_mixed_operation_full_miss_is_no_data(self):
        result = semester_credits_and_prerequisites(
            DB_PATH,
            "PROGRAM_WITHOUT_THIS_PLAN",
            "no_coop",
            2,
            2,
            "99999999",
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(result["plans"], [])

    def test_mixed_operation_excludes_unrelated_plan_provenance(self):
        result = semester_credits_and_prerequisites(
            DB_PATH,
            "IT",
            "no_coop",
            2,
            2,
            "06016420",
        )

        references = result["plans"][0]["prerequisites"][0]["provenance"]
        filenames = {reference["source_filename"] for reference in references}
        self.assertEqual(
            filenames,
            {
                "it_page_034.png",
                "it_page_035.png",
                "it_page_333.png",
                "it_page_334.png",
                "it_page_338.png",
            },
        )
        self.assertNotIn("it_page_328.png", filenames)
        self.assertNotIn("it_page_371.png", filenames)


if __name__ == "__main__":
    unittest.main()
