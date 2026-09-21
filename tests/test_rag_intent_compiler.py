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

    def test_count_query_compiles_to_count_and_preserves_scope(self):
        base = replace(
            parse_query_spec("IT แผนสหกิจ ปี 3 เทอม 1 วิชาเลือก มีกี่วิชา"),
            operations=(),
        )
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "count_query",
                requested_facts=("course_list",),
            ),
            authoritative_program="IT",
            authoritative_plans=("coop",),
            authoritative_years=(3,),
            authoritative_semesters=(1,),
        )

        self.assertEqual(result.operations, ("count",))
        self.assertEqual(result.program, "IT")
        self.assertEqual(result.plans, ("coop",))
        self.assertEqual(result.years, (3,))
        self.assertEqual(result.semesters, (1,))
        self.assertEqual(result.category, "วิชาเลือก")

    def test_count_query_rejects_topic_and_judgement(self):
        cases = (
            replace(parse_query_spec("IT ปี 3 มีวิชาเกี่ยวกับ data กี่วิชา"), operations=()),
            replace(parse_query_spec("IT ปี 3 มีกี่วิชา"), judgement="quantity", operations=()),
        )
        for base in cases:
            with self.subTest(base=base), self.assertRaises(IntentCompilerError):
                compile_intent_to_query_spec(
                    base,
                    self.interpretation(
                        "count_query",
                        requested_facts=("course_list",),
                    ),
                    authoritative_program="IT",
                    authoritative_years=(3,),
                )

    def test_count_query_rejects_comparison_and_multiple_operations(self):
        base = replace(
            parse_query_spec("IT แผนสหกิจกับไม่สหกิจต่างกันยังไง"),
            operations=("count", "compare"),
        )

        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                base,
                self.interpretation(
                    "count_query",
                    requested_facts=("course_list",),
                ),
                authoritative_program="IT",
            )

    def test_count_query_rejects_targets_and_unresolved_fields(self):
        target_base = replace(
            parse_query_spec("IT 06016420 มีกี่วิชา"),
            operations=(),
        )
        unresolved_base = replace(
            parse_query_spec("IT ปี 2 มีกี่วิชา"),
            operations=(),
        )
        cases = (
            (target_base, {"course_codes": ("06016420",)}),
            (unresolved_base, {"unresolved": ("requirement_type",)}),
        )
        for base, fields in cases:
            with self.subTest(fields=fields), self.assertRaises(IntentCompilerError):
                compile_intent_to_query_spec(
                    base,
                    self.interpretation(
                        "count_query",
                        requested_facts=("course_list",),
                        **fields,
                    ),
                    authoritative_program="IT",
                    allowed_course_codes=("06016420",),
                    authoritative_years=(2,),
                )

    def test_count_query_rejects_invented_code_and_scope_widening(self):
        base = replace(parse_query_spec("IT ปี 2 มีกี่วิชา"), operations=())
        cases = (
            {"course_codes": ("06016420",)},
            {"years": (3,)},
        )
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(IntentCompilerError):
                compile_intent_to_query_spec(
                    base,
                    self.interpretation(
                        "count_query",
                        requested_facts=("course_list",),
                        **fields,
                    ),
                    authoritative_program="IT",
                    authoritative_years=(2,),
                )

    def test_qp35_count_mismatches_remain_excluded(self):
        for question in (
            "ปี 2 มีวิชาเกี่ยวกับคอมพิวเตอร์เยอะมั้ย",
            "แผนสหกิจมีวิชาเกี่ยวกับ data มากกว่าแผนปกติไหม",
        ):
            with self.subTest(question=question):
                base = replace(parse_query_spec(question), operations=())
                with self.assertRaises(IntentCompilerError):
                    compile_intent_to_query_spec(
                        base,
                        self.interpretation(
                            "count_query",
                            requested_facts=("course_list",),
                        ),
                        authoritative_program="IT",
                    )

    def test_placement_intent_preserves_authoritative_year_five(self):
        base = replace(
            parse_query_spec("IT ปี 5 06016420 ถามช่วงเรียน"),
            operations=(),
        )
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "placement_query",
                years=(5,),
                course_codes=("06016420",),
                requested_facts=("placement",),
            ),
            authoritative_program="IT",
            authoritative_years=(5,),
            allowed_course_codes=("06016420",),
        )

        self.assertEqual(result.operations, ("placement",))
        self.assertEqual(result.years, (5,))
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

    def test_preference_evidence_maps_to_grounded_topic_list(self):
        base = replace(
            parse_query_spec("IT ปี 3 อยากเน้น data"),
            operations=(),
            topic=None,
        )
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "preference_recommendation_evidence",
                topic="data",
                requested_facts=("course_list",),
                judgement_dimension="preference",
            ),
            authoritative_program="IT",
            authoritative_years=(3,),
        )

        self.assertEqual(result.operations, ("list",))
        self.assertEqual(result.judgement, "preference")
        self.assertEqual(result.topic, "data")
        self.assertEqual(result.years, (3,))

    def test_preference_evidence_with_prerequisite_maps_to_two_operations(self):
        base = replace(
            parse_query_spec("IT ปี 3 อยากเน้น data"),
            operations=(),
            topic=None,
        )
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "preference_recommendation_evidence",
                topic="data",
                requested_facts=("course_list", "prerequisite"),
                judgement_dimension="preference",
            ),
            authoritative_program="IT",
            authoritative_years=(3,),
        )

        self.assertEqual(result.operations, ("list", "prerequisite"))
        self.assertEqual(result.judgement, "preference")
        self.assertEqual(result.topic, "data")
        self.assertEqual(result.program, "IT")
        self.assertEqual(result.years, (3,))

    def test_preference_evidence_rejects_noncanonical_fact_shapes(self):
        base = replace(parse_query_spec("IT อยากเน้น data"), operations=())
        for facts in (
            ("prerequisite",),
            ("prerequisite", "course_list"),
            ("course_list", "prerequisite", "placement"),
            ("course_list", "preference_evidence"),
        ):
            with self.subTest(facts=facts), self.assertRaises(IntentCompilerError):
                compile_intent_to_query_spec(
                    base,
                    self.interpretation(
                        "preference_recommendation_evidence",
                        topic="data",
                        requested_facts=facts,
                        judgement_dimension="preference",
                    ),
                    authoritative_program="IT",
                )

    def test_preference_evidence_merges_only_allowed_base_operations(self):
        for base_operations in ((), ("list",), ("prerequisite",), ("list", "prerequisite")):
            with self.subTest(base_operations=base_operations):
                base = replace(
                    parse_query_spec("IT อยากเน้น data"),
                    operations=base_operations,
                )
                result = compile_intent_to_query_spec(
                    base,
                    self.interpretation(
                        "preference_recommendation_evidence",
                        topic="data",
                        requested_facts=("course_list", "prerequisite"),
                        judgement_dimension="preference",
                    ),
                    authoritative_program="IT",
                )
                self.assertEqual(result.operations, ("list", "prerequisite"))

    def test_preference_evidence_rejects_unrelated_base_operation(self):
        base = replace(parse_query_spec("IT อยากเน้น data"), operations=("describe",))
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                base,
                self.interpretation(
                    "preference_recommendation_evidence",
                    topic="data",
                    requested_facts=("course_list",),
                    judgement_dimension="preference",
                ),
                authoritative_program="IT",
            )

    def test_preference_evidence_requires_topic(self):
        base = replace(parse_query_spec("IT ปี 3 อยากเลือกวิชา"), operations=())
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                base,
                self.interpretation(
                    "preference_recommendation_evidence",
                    requested_facts=("course_list",),
                    judgement_dimension="preference",
                ),
                authoritative_program="IT",
                authoritative_years=(3,),
            )

    def test_preference_evidence_rejects_conflicting_topic(self):
        base = replace(
            parse_query_spec("IT มีวิชาเกี่ยวกับ network อะไรบ้าง"),
            operations=(),
        )
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                base,
                self.interpretation(
                    "preference_recommendation_evidence",
                    topic="data",
                    requested_facts=("course_list",),
                    judgement_dimension="preference",
                ),
                authoritative_program="IT",
            )

    def test_preference_evidence_rejects_non_list_facts(self):
        base = replace(parse_query_spec("IT อยากเน้น data"), operations=())
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                base,
                self.interpretation(
                    "preference_recommendation_evidence",
                    topic="data",
                    requested_facts=("course_description",),
                    judgement_dimension="preference",
                ),
                authoritative_program="IT",
            )

    def test_preference_evidence_preserves_authoritative_plan_and_scope(self):
        base = replace(
            parse_query_spec("IT แผนสหกิจ ปี 3 เทอม 1 อยากเน้น data"),
            operations=(),
            topic=None,
        )
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "preference_recommendation_evidence",
                topic="data",
                requested_facts=("course_list",),
                judgement_dimension="preference",
            ),
            authoritative_program="IT",
            authoritative_plans=("coop",),
            authoritative_years=(3,),
            authoritative_semesters=(1,),
        )

        self.assertEqual(result.program, "IT")
        self.assertEqual(result.plans, ("coop",))
        self.assertEqual(result.years, (3,))
        self.assertEqual(result.semesters, (1,))
        self.assertEqual(result.operations, ("list",))
        self.assertEqual(result.judgement, "preference")

    def test_preference_evidence_does_not_create_ranking_or_new_operation(self):
        base = replace(parse_query_spec("IT อยากเน้น data"), operations=())
        result = compile_intent_to_query_spec(
            base,
            self.interpretation(
                "preference_recommendation_evidence",
                topic="data",
                requested_facts=("course_list",),
                judgement_dimension="preference",
            ),
            authoritative_program="IT",
        )

        self.assertEqual(result.operations, ("list",))
        self.assertNotEqual(result.judgement, "recommendation")
        self.assertNotIn("rank", result.__class__.__annotations__)

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
