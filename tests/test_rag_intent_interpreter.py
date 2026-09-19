import dataclasses
import json
import unittest

from rag.intent_interpreter import (
    ExecutionEligibility,
    IntentInterpretation,
    IntentValidationError,
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


if __name__ == "__main__":
    unittest.main()
