import dataclasses
import json
import unittest
from unittest.mock import patch

from rag.intent_interpreter import (
    INTENTS,
    ExecutionEligibility,
    IntentInterpretation,
    IntentValidationError,
    build_intent_prompt,
    interpret_question_intent,
    parse_intent_payload,
    validate_execution_scope,
)


def payload(**fields):
    return json.dumps(fields, ensure_ascii=False)


class RagIntentInterpreterTest(unittest.TestCase):
    def test_valid_topic_interpretation(self):
        interpretation = parse_intent_payload(
            payload(
                intent="topic_course_search",
                proposed_program="IT",
                proposed_plans=["no_coop"],
                proposed_years=[2],
                topic="database",
                requested_facts=["course_list"],
            )
        )

        self.assertEqual(interpretation.intent, "topic_course_search")
        self.assertEqual(interpretation.proposed_program, "IT")
        self.assertEqual(interpretation.proposed_plans, ("no_coop",))
        self.assertEqual(interpretation.topic, "database")
        self.assertEqual(
            validate_execution_scope(
                interpretation,
                program="IT",
                allowed_plans=("no_coop",),
                allowed_years=(2,),
            ),
            ExecutionEligibility(True, "ok"),
        )

    def test_valid_placement_interpretation(self):
        interpretation = parse_intent_payload(
            payload(
                intent="placement_query",
                proposed_program="IT",
                course_codes=["06016414"],
                requested_facts=["placement"],
            )
        )

        self.assertEqual(interpretation.course_codes, ("06016414",))
        self.assertEqual(
            validate_execution_scope(
                interpretation,
                program="IT",
                allowed_course_codes=("06016414",),
            ),
            ExecutionEligibility(True, "ok"),
        )

    def test_valid_program_discovery_without_program(self):
        interpretation = parse_intent_payload(
            payload(
                intent="program_discovery",
                course_codes=["06016414"],
                requested_facts=["program_identity"],
            )
        )

        self.assertIsNone(interpretation.proposed_program)
        self.assertEqual(
            validate_execution_scope(
                interpretation,
                allowed_course_codes=("06016414",),
            ),
            ExecutionEligibility(True, "ok"),
        )

    def test_unknown_intent_rejected(self):
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(payload(intent="answer_anything"))
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(payload())

    def test_unknown_field_rejected(self):
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(intent="placement_query", someday_field="x")
            )

    def test_sql_and_raw_id_fields_rejected(self):
        for forbidden in (
            {"sql": "SELECT course_id FROM courses"},
            {"course_id": 1},
            {"placement_id": 11},
            {"credits": 3},
            {"provenance": []},
            {"final_answer": "text"},
        ):
            with self.subTest(forbidden=tuple(forbidden)):
                with self.assertRaises(IntentValidationError):
                    parse_intent_payload(
                        payload(intent="placement_query", **forbidden)
                    )

    def test_malformed_json_rejected(self):
        for bad in ("{not json", "[1, 2]", '"just a string"', "42", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(IntentValidationError):
                    parse_intent_payload(bad)
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(None)

    def test_program_conflict_rejected(self):
        interpretation = parse_intent_payload(
            payload(intent="placement_query", proposed_program="DSBA")
        )

        self.assertEqual(
            validate_execution_scope(interpretation, program="IT"),
            ExecutionEligibility(False, "program_conflict"),
        )

    def test_missing_program_cannot_be_invented(self):
        invented = parse_intent_payload(
            payload(
                intent="placement_query",
                proposed_program="IT",
                course_codes=["06016414"],
            )
        )
        self.assertEqual(
            validate_execution_scope(
                invented, allowed_course_codes=("06016414",)
            ),
            ExecutionEligibility(False, "program_unsubstantiated"),
        )

        absent = parse_intent_payload(payload(intent="placement_query"))
        self.assertEqual(
            validate_execution_scope(absent),
            ExecutionEligibility(False, "program_missing"),
        )

    def test_invalid_year_semester_rejected(self):
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(intent="placement_query", proposed_years=["second"])
            )
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(intent="placement_query", proposed_years=[True])
            )

        year = parse_intent_payload(
            payload(
                intent="placement_query",
                proposed_program="IT",
                proposed_years=[9],
            )
        )
        self.assertEqual(
            validate_execution_scope(
                year, program="IT", allowed_years=(9,)
            ),
            ExecutionEligibility(False, "year_out_of_range"),
        )
        semester = parse_intent_payload(
            payload(
                intent="placement_query",
                proposed_program="IT",
                proposed_semesters=[3],
            )
        )
        self.assertEqual(
            validate_execution_scope(
                semester, program="IT", allowed_semesters=(3,)
            ),
            ExecutionEligibility(False, "semester_out_of_range"),
        )
        mismatch = parse_intent_payload(
            payload(
                intent="placement_query",
                proposed_program="IT",
                proposed_years=[3],
            )
        )
        self.assertEqual(
            validate_execution_scope(
                mismatch, program="IT", allowed_years=(2,)
            ),
            ExecutionEligibility(False, "year_conflict"),
        )

    def test_course_code_absent_from_allowed_refs_rejected(self):
        interpretation = parse_intent_payload(
            payload(
                intent="placement_query",
                proposed_program="IT",
                course_codes=["06016414"],
            )
        )

        self.assertEqual(
            validate_execution_scope(
                interpretation, program="IT", allowed_course_codes=("06019999",)
            ),
            ExecutionEligibility(False, "course_code_not_allowed"),
        )
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(intent="placement_query", course_codes=["IT-123"])
            )

    def test_duplicate_values_rejected(self):
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(
                    intent="plan_comparison",
                    proposed_plans=["coop", "coop"],
                )
            )
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(
                    intent="course_comparison",
                    course_codes=["06016414", "06016414"],
                )
            )

    def test_nonempty_unresolved_invalid_for_execution(self):
        interpretation = parse_intent_payload(
            payload(
                intent="topic_course_search",
                proposed_program="IT",
                topic="database",
                unresolved=["requirement_type"],
            )
        )

        self.assertEqual(interpretation.unresolved, ("requirement_type",))
        self.assertEqual(
            validate_execution_scope(interpretation, program="IT"),
            ExecutionEligibility(False, "unresolved_items_present"),
        )

    def test_unsupported_judgement_dimension_rejected(self):
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(
                    intent="workload_judgement",
                    judgement_dimension="difficulty",
                )
            )
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(
                    intent="preference_recommendation_evidence",
                    judgement_dimension="workload",
                )
            )
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(
                    intent="placement_query",
                    judgement_dimension="workload",
                )
            )

        interpretation = parse_intent_payload(
            payload(
                intent="workload_judgement",
                proposed_program="IT",
                judgement_dimension="workload",
                requested_facts=["workload_evidence"],
            )
        )
        self.assertEqual(interpretation.judgement_dimension, "workload")
        self.assertEqual(
            validate_execution_scope(interpretation, program="IT"),
            ExecutionEligibility(True, "ok"),
        )

    def test_frozen_immutable_structure(self):
        interpretation = parse_intent_payload(
            payload(intent="program_discovery")
        )

        self.assertIsInstance(interpretation, IntentInterpretation)
        for field in dataclasses.fields(interpretation):
            value = getattr(interpretation, field.name)
            if field.name in (
                "proposed_program",
                "topic",
                "judgement_dimension",
            ):
                self.assertIsNone(value)
            elif field.name == "intent":
                self.assertEqual(value, "program_discovery")
            else:
                self.assertIsInstance(value, tuple)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            interpretation.intent = "placement_query"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            validate_execution_scope(interpretation).eligible = True
        self.assertEqual(hash(interpretation), hash(interpretation))

    def test_empty_values_normalized(self):
        interpretation = parse_intent_payload(
            payload(
                intent="topic_course_search",
                proposed_program="  ",
                topic="",
                judgement_dimension=None,
            )
        )

        self.assertIsNone(interpretation.proposed_program)
        self.assertIsNone(interpretation.topic)
        self.assertIsNone(interpretation.judgement_dimension)
        self.assertEqual(interpretation.proposed_plans, ())

    def test_oversized_and_wrong_types_rejected(self):
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(intent="topic_course_search", topic="x" * 65)
            )
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(
                    intent="topic_course_search",
                    proposed_plans=["a", "b", "c", "d", "e"],
                )
            )
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(intent="topic_course_search", proposed_plans="coop")
            )
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(
                    intent="topic_course_search",
                    requested_facts=["best_courses"],
                )
            )


    def test_interpret_valid_topic_json(self):
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return payload(
                intent="topic_course_search",
                proposed_program="IT",
                topic="database",
                requested_facts=["course_list"],
            )

        with patch(
            "rag.intent_interpreter.validate_execution_scope",
            side_effect=AssertionError("scope validation belongs elsewhere"),
        ):
            interpretation = interpret_question_intent(
                "IT มีวิชาเกี่ยวกับ database อะไรบ้าง", fake_model
            )

        self.assertEqual(interpretation.intent, "topic_course_search")
        self.assertEqual(interpretation.proposed_program, "IT")
        self.assertEqual(interpretation.topic, "database")
        self.assertEqual(len(calls), 1)

    def test_interpret_valid_placement_json(self):
        def fake_model(prompt):
            return payload(
                intent="placement_query",
                proposed_program="IT",
                course_codes=["06016414"],
                requested_facts=["placement"],
            )

        interpretation = interpret_question_intent(
            "IT 06016414 เรียนปีไหน", fake_model
        )

        self.assertEqual(interpretation.intent, "placement_query")
        self.assertEqual(interpretation.course_codes, ("06016414",))

    def test_interpret_valid_program_discovery_without_program(self):
        def fake_model(prompt):
            return payload(
                intent="program_discovery",
                course_codes=["06016414"],
                requested_facts=["program_identity"],
            )

        interpretation = interpret_question_intent(
            "06016414 อยู่ในหลักสูตรไหน", fake_model
        )

        self.assertIsNone(interpretation.proposed_program)
        self.assertEqual(
            validate_execution_scope(
                interpretation, allowed_course_codes=("06016414",)
            ),
            ExecutionEligibility(True, "ok"),
        )

    def test_prompt_states_json_only_no_answer_no_sql_no_inference(self):
        prompt = build_intent_prompt("IT 06016414 เรียนปีไหน")

        self.assertIn("JSON object", prompt)
        self.assertIn("No Markdown", prompt)
        self.assertIn("code fence", prompt.casefold())
        self.assertIn("do not answer", prompt.casefold())
        self.assertIn("SQL", prompt)
        self.assertIn("Never infer", prompt)
        self.assertIn("course-code prefixes", prompt)

    def test_prompt_contains_exact_wire_keys(self):
        prompt = build_intent_prompt("IT 06016414 เรียนปีไหน")

        for key in (
            "intent",
            "proposed_program",
            "proposed_plans",
            "proposed_years",
            "proposed_semesters",
            "course_codes",
            "topic",
            "requested_facts",
            "judgement_dimension",
            "unresolved",
        ):
            self.assertIn(key, prompt)

    def test_prompt_exposes_bounded_intent_allowlist(self):
        prompt = build_intent_prompt("IT 06016414 เรียนปีไหน")

        for intent in INTENTS:
            self.assertIn(intent, prompt)

    def test_model_called_exactly_once(self):
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return payload(intent="program_discovery")

        interpret_question_intent("06016414 อยู่ในหลักสูตรไหน", fake_model)

        self.assertEqual(len(calls), 1)

    def test_malformed_json_fails_after_single_call(self):
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return "{not json"

        with self.assertRaises(IntentValidationError):
            interpret_question_intent("IT 06016414 เรียนปีไหน", fake_model)
        self.assertEqual(len(calls), 1)

    def test_fenced_json_rejected_without_stripping(self):
        def fake_model(prompt):
            return (
                "```json\n"
                + payload(intent="program_discovery")
                + "\n```"
            )

        with self.assertRaises(IntentValidationError):
            interpret_question_intent("06016414 อยู่ในหลักสูตรไหน", fake_model)

    def test_unknown_field_rejected_via_interpret(self):
        def fake_model(prompt):
            return payload(intent="program_discovery", extra="x")

        with self.assertRaises(IntentValidationError):
            interpret_question_intent("06016414 อยู่ในหลักสูตรไหน", fake_model)

    def test_model_non_string_output_rejected(self):
        with self.assertRaises(TypeError):
            interpret_question_intent(
                "IT 06016414 เรียนปีไหน", lambda prompt: {"intent": "x"}
            )

    def test_empty_question_rejected_before_model_call(self):
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return payload(intent="program_discovery")

        for bad in ("", "   ", None, 42):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    interpret_question_intent(bad, fake_model)
        self.assertEqual(calls, [])

    def test_non_callable_model_rejected(self):
        with self.assertRaises(TypeError):
            interpret_question_intent("IT 06016414 เรียนปีไหน", "not callable")

    def test_model_exception_propagates_without_retry(self):
        calls = []

        def failing_model(prompt):
            calls.append(prompt)
            raise RuntimeError("provider unavailable")

        with self.assertRaises(RuntimeError):
            interpret_question_intent("IT 06016414 เรียนปีไหน", failing_model)
        self.assertEqual(len(calls), 1)

    def test_prompt_has_no_schema_or_sql_generation(self):
        prompt = build_intent_prompt("IT 06016414 เรียนปีไหน")

        self.assertNotIn("SELECT", prompt)
        self.assertNotIn("CREATE TABLE", prompt)
        self.assertNotIn("course_id", prompt)
        self.assertNotIn("schema", prompt.casefold())
        self.assertNotIn("generate SQL", prompt.casefold())


    def test_prompt_states_bare_8_digit_course_codes(self):
        prompt = build_intent_prompt("IT 06016414 เรียนปีไหน")

        self.assertIn("ONLY a bare exact", prompt)
        self.assertIn("8-digit ASCII course code", prompt)
        self.assertIn("never include spaces", prompt.casefold())

    def test_prompt_shows_program_code_separation_example(self):
        prompt = build_intent_prompt("IT 06016414 เรียนปีไหน")

        self.assertIn('"proposed_program": "IT"', prompt)
        self.assertIn('"course_codes": ["06016414"]', prompt)
        self.assertIn("separate semantic fields", prompt)

    def test_prompt_shows_incorrect_prefixed_code_example(self):
        prompt = build_intent_prompt("IT 06016414 เรียนปีไหน")

        self.assertIn('"course_codes": ["IT 06016414"]', prompt)
        self.assertIn("INCORRECT", prompt)

    def test_prompt_shows_two_course_example(self):
        prompt = build_intent_prompt("IT 06016414 เรียนปีไหน")

        self.assertIn('"course_codes": ["06016414", "06016420"]', prompt)

    def test_parser_still_rejects_program_prefixed_code(self):
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(
                    intent="placement_query",
                    proposed_program="IT",
                    course_codes=["IT 06016414"],
                )
            )

    def test_parser_still_rejects_thai_prefixed_code(self):
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(
                payload(
                    intent="placement_query",
                    course_codes=["วิชา 06016414"],
                )
            )

    def test_parser_still_accepts_bare_single_code(self):
        interpretation = parse_intent_payload(
            payload(
                intent="placement_query",
                proposed_program="IT",
                course_codes=["06016414"],
            )
        )

        self.assertEqual(interpretation.course_codes, ("06016414",))

    def test_parser_still_accepts_bare_code_pair(self):
        interpretation = parse_intent_payload(
            payload(
                intent="course_comparison",
                proposed_program="IT",
                course_codes=["06016414", "06016420"],
            )
        )

        self.assertEqual(
            interpretation.course_codes, ("06016414", "06016420")
        )


    def test_prompt_requires_course_list_for_preference_intent(self):
        prompt = build_intent_prompt("IT ปี 3 อยากเรียน data")

        self.assertIn("preference_recommendation_evidence", prompt)
        self.assertIn('MUST be exactly ["course_list"]', prompt)

    def test_prompt_forbids_preference_evidence_fact_for_preference(self):
        prompt = build_intent_prompt("IT ปี 3 อยากเรียน data")

        self.assertIn('Do NOT emit ["preference_evidence"]', prompt)

    def test_prompt_contains_preference_worked_example(self):
        prompt = build_intent_prompt("IT ปี 3 อยากเรียน data")

        self.assertIn("ถ้าอยากปูทางไป data", prompt)
        self.assertIn('"topic": "data"', prompt)
        self.assertIn('"requested_facts": ["course_list"]', prompt)
        self.assertIn('WRONG "requested_facts": ["preference_evidence"]', prompt)

    def test_prompt_preserves_preference_dimension_rule(self):
        prompt = build_intent_prompt("IT ปี 3 อยากเรียน data")

        self.assertIn("judgement_dimension MUST be preference", prompt)

    def test_parser_still_accepts_preference_evidence_fact_shape(self):
        interpretation = parse_intent_payload(
            payload(
                intent="preference_recommendation_evidence",
                proposed_program="IT",
                topic="data",
                requested_facts=["preference_evidence"],
                judgement_dimension="preference",
            )
        )

        self.assertEqual(
            interpretation.requested_facts, ("preference_evidence",)
        )

    def test_parser_accepts_course_list_preference_shape(self):
        interpretation = parse_intent_payload(
            payload(
                intent="preference_recommendation_evidence",
                proposed_program="IT",
                proposed_years=[3],
                topic="data",
                requested_facts=["course_list"],
                judgement_dimension="preference",
            )
        )

        self.assertEqual(interpretation.requested_facts, ("course_list",))
        self.assertEqual(
            validate_execution_scope(
                interpretation, program="IT", allowed_years=(3,)
            ),
            ExecutionEligibility(True, "ok"),
        )


    def test_compiler_behavior_untouched_for_preference_shapes(self):
        from rag.intent_compiler import (
            IntentCompilerError,
            compile_intent_to_query_spec,
        )
        from rag.query_spec import parse_query_spec

        base = parse_query_spec("IT ปี 3 ถ้าอยากปูทางไป data")
        course_list_shape = parse_intent_payload(
            payload(
                intent="preference_recommendation_evidence",
                proposed_program="IT",
                proposed_years=[3],
                topic="data",
                requested_facts=["course_list"],
                judgement_dimension="preference",
            )
        )
        compiled = compile_intent_to_query_spec(
            base,
            course_list_shape,
            authoritative_program="IT",
            authoritative_years=(3,),
        )
        self.assertEqual(compiled.operations, ("list",))
        self.assertEqual(compiled.judgement, "preference")

        evidence_only_shape = parse_intent_payload(
            payload(
                intent="preference_recommendation_evidence",
                proposed_program="IT",
                topic="data",
                requested_facts=["preference_evidence"],
                judgement_dimension="preference",
            )
        )
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                base,
                evidence_only_shape,
                authoritative_program="IT",
            )


if __name__ == "__main__":
    unittest.main()
