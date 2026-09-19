import unittest
from dataclasses import replace

from rag.intent_compiler import (
    IntentCompilerError,
    compile_intent_to_query_spec,
)
from rag.intent_interpreter import IntentInterpretation
from rag.query_spec import parse_query_spec


class IntentCompilerTest(unittest.TestCase):
    @staticmethod
    def interpretation(
        intent,
        *,
        program=None,
        plans=(),
        years=(),
        semesters=(),
        course_codes=(),
        topic=None,
        requested_facts=(),
        judgement_dimension=None,
        unresolved=(),
    ):
        return IntentInterpretation(
            intent=intent,
            proposed_program=program,
            proposed_plans=tuple(plans),
            proposed_years=tuple(years),
            proposed_semesters=tuple(semesters),
            course_codes=tuple(course_codes),
            topic=topic,
            requested_facts=tuple(requested_facts),
            judgement_dimension=judgement_dimension,
            unresolved=tuple(unresolved),
        )

    def test_topic_intent_fills_missing_topic(self):
        base = replace(
            parse_query_spec("IT ปี 2 มีวิชาอะไรบ้าง"),
            operations=("list",),
            topic=None,
        )
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "topic_course_search",
                topic="database",
                requested_facts=("course_list",),
            ),
            authoritative_program="IT",
            authoritative_years=(2,),
        )

        self.assertEqual(result.operations, ("list",))
        self.assertEqual(result.topic, "database")
        self.assertEqual(result.years, (2,))

    def test_placement_intent_maps_to_existing_operation(self):
        base = replace(
            parse_query_spec("IT 06016420 ถามช่วงเรียน"),
            operations=(),
        )
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "placement_query",
                course_codes=("06016420",),
                requested_facts=("placement",),
            ),
            authoritative_program="IT",
            allowed_course_codes=("06016420",),
        )

        self.assertEqual(result.operations, ("placement",))
        self.assertEqual(result.course_codes, ("06016420",))

    def test_prerequisite_intent_maps_to_existing_operation(self):
        base = replace(
            parse_query_spec("IT 06016420 ถามวิชาก่อนหน้า"),
            operations=(),
        )
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "prerequisite_query",
                course_codes=("06016420",),
                requested_facts=("prerequisite",),
            ),
            authoritative_program="IT",
            allowed_course_codes=("06016420",),
        )

        self.assertEqual(result.operations, ("prerequisite",))

    def test_similarity_requires_two_codes_and_course_group(self):
        base = replace(
            parse_query_spec("IT 06016420 กับ 06016421 ถามเนื้อหา"),
            operations=(),
            group_by=(),
        )
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "similarity_query",
                course_codes=("06016420", "06016421"),
                requested_facts=("similarity",),
            ),
            authoritative_program="IT",
            allowed_course_codes=("06016420", "06016421"),
        )

        self.assertEqual(result.operations, ("similarity",))
        self.assertEqual(result.group_by, ("course",))

    def test_plan_comparison_requires_two_authorized_plans(self):
        base = replace(
            parse_query_spec("IT เปรียบเทียบแผน"),
            operations=(),
            plans=(),
            group_by=(),
        )
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "plan_comparison",
                plans=("coop", "no_coop"),
                requested_facts=("plan_comparison",),
            ),
            authoritative_program="IT",
            authoritative_plans=("coop", "no_coop"),
        )

        self.assertEqual(result.operations, ("compare",))
        self.assertEqual(result.plans, ("coop", "no_coop"))
        self.assertEqual(result.group_by, ("plan",))

    def test_program_discovery_is_valid_without_program(self):
        base = replace(
            parse_query_spec("06016420 อยู่ในหลักสูตรอะไร"),
            program=None,
            operations=(),
        )
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "program_discovery",
                course_codes=("06016420",),
                requested_facts=("program_identity",),
            ),
            allowed_course_codes=("06016420",),
        )

        self.assertEqual(result.operations, ("program_discovery",))
        self.assertIsNone(result.program)

    def test_deterministic_base_program_is_preserved(self):
        base = replace(parse_query_spec("IT 06016420"), operations=())
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "course_description",
                course_codes=("06016420",),
                requested_facts=("course_description",),
            ),
            allowed_course_codes=("06016420",),
        )

        self.assertEqual(result.program, "IT")

    def test_interpretation_cannot_override_base_program(self):
        base = replace(parse_query_spec("IT 06016420"), operations=())
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                base,
                self.interpretation(
                    "course_description",
                    program="DSBA",
                    course_codes=("06016420",),
                    requested_facts=("course_description",),
                ),
                allowed_course_codes=("06016420",),
            )

    def test_deterministic_year_and_semester_are_preserved(self):
        base = replace(
            parse_query_spec("IT ปี 2 เทอม 1 มีวิชาอะไรบ้าง"),
            operations=("list",),
        )
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "topic_course_search",
                topic="database",
                requested_facts=("course_list",),
            ),
            authoritative_program="IT",
            authoritative_years=(2,),
            authoritative_semesters=(1,),
        )

        self.assertEqual(result.years, (2,))
        self.assertEqual(result.semesters, (1,))

    def test_category_is_preserved_and_interpretation_cannot_introduce_one(self):
        base = replace(
            parse_query_spec("IT ปี 2 วิชาเลือกมีอะไรบ้าง"),
            operations=("list",),
        )
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "topic_course_search",
                topic="database",
                requested_facts=("course_list",),
            ),
            authoritative_program="IT",
            authoritative_years=(2,),
        )

        self.assertEqual(result.category, base.category)
        self.assertIsNone(
            getattr(self.interpretation("topic_course_search"), "category", None)
        )

    def test_course_code_widening_is_rejected(self):
        base = replace(parse_query_spec("IT 06016420"), operations=())
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                base,
                self.interpretation(
                    "similarity_query",
                    course_codes=("06016420", "06016421"),
                    requested_facts=("similarity",),
                ),
                allowed_course_codes=("06016420", "06016421"),
            )

    def test_conflicting_topic_is_rejected(self):
        base = replace(
            parse_query_spec("IT มีวิชาเกี่ยวกับ database อะไรบ้าง"),
            operations=("list",),
        )
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                base,
                self.interpretation(
                    "topic_course_search",
                    topic="network",
                    requested_facts=("course_list",),
                ),
                authoritative_program="IT",
            )

    def test_unsupported_requested_fact_is_rejected(self):
        base = replace(parse_query_spec("IT 06016420"), operations=())
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                base,
                self.interpretation(
                    "placement_query",
                    course_codes=("06016420",),
                    requested_facts=("course_list",),
                ),
                allowed_course_codes=("06016420",),
            )

    def test_unresolved_interpretation_is_rejected(self):
        base = replace(parse_query_spec("IT 06016420"), operations=())
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                base,
                self.interpretation(
                    "placement_query",
                    course_codes=("06016420",),
                    requested_facts=("placement",),
                    unresolved=("which plan",),
                ),
                allowed_course_codes=("06016420",),
            )

    def test_invalid_workload_and_preference_without_evidence_fail_closed(self):
        base = replace(parse_query_spec("IT 06016420"), operations=())
        for intent, dimension in (
            ("workload_judgement", "workload"),
            ("preference_recommendation_evidence", "preference"),
        ):
            with self.subTest(intent=intent), self.assertRaises(IntentCompilerError):
                compile_intent_to_query_spec(
                    base,
                    self.interpretation(
                        intent,
                        course_codes=("06016420",),
                        requested_facts=(f"{dimension}_evidence",),
                        judgement_dimension=dimension,
                    ),
                    allowed_course_codes=("06016420",),
                )

    def test_original_questions_are_preserved(self):
        base = replace(parse_query_spec("IT 06016420"), operations=())
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "course_description",
                course_codes=("06016420",),
                requested_facts=("course_description",),
            ),
            allowed_course_codes=("06016420",),
        )

        self.assertEqual(result.original_question, base.original_question)
        self.assertEqual(result.normalized_question, base.normalized_question)

    def test_compilation_does_not_mutate_inputs(self):
        base = parse_query_spec("IT 06016420")
        interpretation = self.interpretation(
            "course_description",
            course_codes=("06016420",),
            requested_facts=("course_description",),
        )
        result = compile_intent_to_query_spec(
            replace(base, operations=()),
            interpretation,
            allowed_course_codes=("06016420",),
        )

        self.assertIsNot(result, base)
        self.assertEqual(base.operations, ())
        self.assertEqual(interpretation.course_codes, ("06016420",))

    def test_existing_query_spec_passes_without_losing_scope(self):
        base = parse_query_spec("IT ปี 2 เทอม 1 มีวิชาเกี่ยวกับ database อะไรบ้าง")
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "topic_course_search",
                requested_facts=("course_list",),
            ),
            authoritative_program="IT",
            authoritative_years=(2,),
            authoritative_semesters=(1,),
        )

        self.assertEqual(result.program, base.program)
        self.assertEqual(result.plans, base.plans)
        self.assertEqual(result.years, base.years)
        self.assertEqual(result.semesters, base.semesters)
        self.assertEqual(result.topic, base.topic)
        self.assertEqual(result.operations, base.operations)


if __name__ == "__main__":
    unittest.main()
