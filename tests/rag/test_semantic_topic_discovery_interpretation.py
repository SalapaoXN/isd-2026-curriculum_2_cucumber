"""Interpretation-contract checks for course-topic discovery and exact targets."""

import json
import unittest

from rag.semantic.interpreter import interpret_semantic_intent
from rag.semantic.prompts import build_semantic_interpreter_prompt
from rag.semantic.compiler import compile_resolved_intent_to_query_spec
from rag.semantic.context import merge_semantic_context
from rag.semantic.schema import ResolvedIntent, ResolvedScope
from rag.semantic.validation import validate_semantic_intent


def _intent_payload(
    *, task, target, scope=None, filters=(), requested_fields=(), clarification=None
):
    return json.dumps(
        {
            "task": task,
            "subject": "course",
            "relation": "credits" if task == "lookup" else None,
            "target": {
                "kind": target[0],
                "raw_text": target[1],
                "normalized_hint": None,
                "ordinal": None,
            },
            "scope": {
                "program": None,
                "catalog": None,
                "plan": None,
                "plan_hint": None,
                "year": None,
                "semester": None,
                **(scope or {}),
            },
            "filters": list(filters),
            "aggregation": None,
            "ranking": None,
            "comparison": None,
            "requested_fields": list(requested_fields),
            "clarification": clarification,
            "policy_topic": None,
            "observed_value": None,
        },
        ensure_ascii=False,
    )


class SemanticTopicDiscoveryInterpretationTests(unittest.TestCase):
    def _interpret(self, question, payload):
        intent, _ = interpret_semantic_intent(question, lambda _prompt: payload)
        validation = validate_semantic_intent(intent, question)
        return intent, validation

    def test_topic_discovery_is_list_related_to_with_no_course_target(self):
        question = "มีวิชาเกี่ยวกับ cyber security อะไรบ้าง"
        topic = "cyber security"
        payload = _intent_payload(
            task="list",
            target=("none", None),
            filters=({"field": "topic", "operator": "related_to", "value": topic},),
            requested_fields=("code", "name"),
        )
        intent, validation = self._interpret(question, payload)

        self.assertEqual(intent.task, "list")
        self.assertEqual(intent.subject, "course")
        self.assertEqual(intent.target.kind, "none")
        self.assertEqual(
            [(item.field, item.operator, item.value) for item in intent.filters],
            [("topic", "related_to", topic)],
        )
        self.assertTrue(validation.valid, validation.reason)

        resolved = ResolvedIntent(
            intent=intent,
            scope=ResolvedScope(program="IT", catalog_key="it-2565"),
        )
        query_spec = compile_resolved_intent_to_query_spec(resolved, question)
        self.assertEqual(query_spec.operations, ("list",))
        self.assertEqual(query_spec.topic, topic)
        self.assertEqual(query_spec.course_codes, ())
        self.assertIsNone(query_spec.course_name)

    def test_unseen_topic_phrase_uses_the_same_discovery_shape(self):
        question = "หา course เกี่ยวกับ marine robotics มีอะไรบ้าง"
        topic = "marine robotics"
        intent, validation = self._interpret(
            question,
            _intent_payload(
                task="list",
                target=("none", None),
                filters=({"field": "topic", "operator": "related_to", "value": topic},),
            ),
        )
        self.assertEqual(intent.task, "list")
        self.assertEqual(intent.target.kind, "none")
        self.assertEqual(intent.filters[0].value, topic)
        self.assertTrue(validation.valid, validation.reason)

    def test_explicit_scope_stays_scope_during_topic_discovery(self):
        question = "IT ปี 3 มีวิชาเกี่ยวกับ marine robotics อะไรบ้าง"
        intent, validation = self._interpret(
            question,
            _intent_payload(
                task="list",
                target=("none", None),
                scope={"program": "IT", "year": 3},
                filters=({"field": "topic", "operator": "related_to", "value": "marine robotics"},),
            ),
        )
        self.assertEqual(intent.scope.program, "IT")
        self.assertEqual(intent.scope.year, 3)
        self.assertEqual(intent.target.kind, "none")
        self.assertEqual(
            [(item.field, item.operator) for item in intent.filters],
            [("topic", "related_to")],
        )
        self.assertTrue(validation.valid, validation.reason)

    def test_named_course_credit_lookup_remains_an_exact_target(self):
        question = "วิชา Cyber Security กี่หน่วย"
        intent, validation = self._interpret(
            question,
            _intent_payload(
                task="lookup",
                target=("literal", "Cyber Security"),
                requested_fields=("credits",),
            ),
        )
        self.assertEqual(intent.task, "lookup")
        self.assertEqual(intent.relation, "credits")
        self.assertEqual(intent.target.kind, "literal")
        self.assertEqual(intent.target.raw_text, "Cyber Security")
        self.assertEqual(intent.filters, ())
        self.assertTrue(validation.valid, validation.reason)

    def test_explicit_course_code_lookup_remains_unchanged(self):
        question = "IT 06016405 กี่หน่วยกิต"
        intent, validation = self._interpret(
            question,
            _intent_payload(
                task="lookup",
                target=("literal", "06016405"),
                scope={"program": "IT"},
                requested_fields=("credits",),
            ),
        )
        self.assertEqual(intent.target.raw_text, "06016405")
        self.assertEqual(intent.filters, ())
        self.assertTrue(validation.valid, validation.reason)

    def test_category_filter_is_not_rewritten_as_topic(self):
        question = "มีวิชาเลือกอะไรบ้าง"
        intent, validation = self._interpret(
            question,
            _intent_payload(
                task="list",
                target=("none", None),
                filters=({"field": "category", "operator": "eq", "value": "วิชาเลือก"},),
            ),
        )
        self.assertEqual(
            [(item.field, item.operator) for item in intent.filters],
            [("category", "eq")],
        )
        self.assertTrue(validation.valid, validation.reason)

    def test_standalone_gened_identifier_is_program_scope(self):
        question = "GENED มีวิชาอะไรบ้าง"
        intent, validation = self._interpret(
            question,
            _intent_payload(
                task="list",
                target=("none", None),
                scope={"program": "GENED"},
                requested_fields=("code", "name"),
            ),
        )
        self.assertTrue(validation.valid, validation.reason)
        self.assertEqual(intent.scope.program, "GENED")
        self.assertEqual(intent.target.kind, "none")
        self.assertEqual(intent.filters, ())

    def test_general_education_group_inside_explicit_program_is_category(self):
        category_filter = {
            "field": "category",
            "operator": "eq",
            "value": "หมวดวิชาศึกษาทั่วไป",
        }
        cases = (
            (
                "IT มีวิชา GENED อะไรบ้าง",
                {"program": "IT"},
            ),
            (
                "วิชา GENED ของ DSBA มีอะไรบ้าง",
                {"program": "DSBA"},
            ),
        )
        for question, scope in cases:
            with self.subTest(question=question):
                intent, validation = self._interpret(
                    question,
                    _intent_payload(
                        task="list",
                        target=("none", None),
                        scope=scope,
                        filters=(category_filter,),
                    ),
                )
                self.assertTrue(validation.valid, validation.reason)
                self.assertEqual(intent.scope.program, scope["program"])
                self.assertEqual(intent.target.kind, "none")
                self.assertEqual(
                    [(item.field, item.operator, item.value) for item in intent.filters],
                    [("category", "eq", "หมวดวิชาศึกษาทั่วไป")],
                )

    def test_group_followup_retains_validated_program_context(self):
        question = "แล้ววิชา gened ล่ะ"
        intent, validation = self._interpret(
            question,
            _intent_payload(
                task="list",
                target=("none", None),
                filters=(
                    {
                        "field": "category",
                        "operator": "eq",
                        "value": "หมวดวิชาศึกษาทั่วไป",
                    },
                ),
                requested_fields=("code", "name"),
            ),
        )
        self.assertTrue(validation.valid, validation.reason)
        self.assertIsNone(intent.scope.program)
        merged = merge_semantic_context(
            intent,
            {"program": "DSBA", "catalog_key": "dsba-2565"},
        )
        self.assertTrue(merged.valid, merged.reason)
        self.assertEqual(merged.program, "DSBA")
        self.assertEqual(merged.catalog_key, "dsba-2565")

    def test_explicit_curriculum_phrase_selects_program_not_category(self):
        question = "หลักสูตร GENED มีวิชาอะไรบ้าง"
        intent, validation = self._interpret(
            question,
            _intent_payload(
                task="list",
                target=("none", None),
                scope={"program": "GENED"},
                requested_fields=("code", "name"),
            ),
        )
        self.assertTrue(validation.valid, validation.reason)
        self.assertEqual(intent.scope.program, "GENED")
        self.assertEqual(intent.filters, ())

    def test_topic_plus_prerequisite_compound_is_explicitly_unsupported(self):
        question = "มีวิชาเกี่ยวกับ marine robotics ที่มีวิชาบังคับก่อนอะไรบ้าง"
        intent, validation = self._interpret(
            question,
            _intent_payload(
                task="unknown",
                target=("none", None),
                filters=(),
                requested_fields=("code", "name", "prerequisites"),
                clarification="This combined topic and prerequisite filter is unsupported.",
            ),
        )
        self.assertEqual(intent.task, "unknown")
        self.assertEqual(intent.filters, ())
        self.assertIsNotNone(intent.clarification)
        self.assertTrue(validation.valid, validation.reason)

    def test_prompt_defines_generic_topic_discovery_and_compound_boundary(self):
        prompt = build_semantic_interpreter_prompt(
            "a generic topic question",
            canonical_program_codes=("GENED", "IT"),
            canonical_category_labels=("หมวดวิชาศึกษาทั่วไป",),
        )
        self.assertIn('target.kind "none"', prompt)
        self.assertIn('operator:"related_to"', prompt)
        self.assertIn("does not support combining topic discovery", prompt)
        self.assertIn("leading standalone curriculum/program identifier", prompt)
        self.assertIn("even if that identifier can also name a course group", prompt)
        self.assertIn("leading uppercase acronym/code", prompt)
        self.assertIn("CANONICAL PROGRAM-CODE CANDIDATES", prompt)
        self.assertIn("GENED, IT", prompt)
        self.assertIn("CANONICAL PLACEMENT CATEGORY LABELS", prompt)
        self.assertIn("หมวดวิชาศึกษาทั่วไป", prompt)
        self.assertIn("leading standalone curriculum/program identifier", prompt)
        self.assertIn("หมวดวิชาศึกษาทั่วไป", prompt)


if __name__ == "__main__":
    unittest.main()
