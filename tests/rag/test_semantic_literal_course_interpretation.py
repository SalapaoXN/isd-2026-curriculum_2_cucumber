"""Deterministic contract tests for literal course attributes vs discovery."""

import json
import unittest

from rag.semantic.interpreter import interpret_semantic_intent
from rag.semantic.prompts import (
    SEMANTIC_INTERPRETER_PROMPT_VERSION,
    build_semantic_interpreter_prompt,
)
from rag.semantic.validation import validate_semantic_intent


def _payload(*, task, relation, target_kind, target_text, filters=(), requested=()):
    return json.dumps(
        {
            "task": task,
            "subject": "course",
            "relation": relation,
            "target": {
                "kind": target_kind,
                "raw_text": target_text,
                "normalized_hint": None,
                "ordinal": None,
            },
            "scope": {
                "program": None,
                "catalog": None,
                "plan": None,
                "year": None,
                "semester": None,
            },
            "filters": list(filters),
            "aggregation": None,
            "ranking": None,
            "comparison": None,
            "requested_fields": list(requested),
            "clarification": None,
            "policy_topic": None,
            "observed_value": None,
        },
        ensure_ascii=False,
    )


class SemanticLiteralCourseInterpretationTests(unittest.TestCase):
    def test_literal_course_attributes_use_literal_target_without_topic_filter(self):
        cases = (
            ("data pipeline เรียนเกี่ยวกับอะไร", "data pipeline", "description", "description"),
            ("วิชา calculus เรียนตอนไหน", "calculus", "placement", "placement"),
            ("network เรียนปีไหน", "network", "placement", "placement"),
            ("Calculus 2 กี่หน่วยกิต", "Calculus 2", "credits", "credits"),
            ("Probability and Statistics ต้องผ่านอะไร", "Probability and Statistics", "prerequisite", "prerequisites"),
        )
        for question, course_text, relation, requested in cases:
            with self.subTest(question=question):
                intent, _ = interpret_semantic_intent(
                    question,
                    lambda _prompt, course_text=course_text, relation=relation, requested=requested: _payload(
                        task="lookup",
                        relation=relation,
                        target_kind="literal",
                        target_text=course_text,
                        requested=(requested,),
                    ),
                )
                validation = validate_semantic_intent(intent, question)

                self.assertTrue(validation.valid, validation.reason)
                self.assertEqual((intent.task, intent.subject, intent.relation),
                                 ("lookup", "course", relation))
                self.assertEqual((intent.target.kind, intent.target.raw_text),
                                 ("literal", course_text))
                self.assertEqual(intent.filters, ())
                self.assertEqual(intent.requested_fields, (requested,))

    def test_course_discovery_requests_use_target_none_and_topic_filter(self):
        cases = (
            ("มีวิชาเกี่ยวกับ data pipeline อะไรบ้าง", "data pipeline"),
            ("มีวิชาเกี่ยวกับ network อะไรบ้าง", "network"),
            ("วิชาเกี่ยวกับ cybersecurity มีอะไร", "cybersecurity"),
        )
        for question, topic in cases:
            with self.subTest(question=question):
                intent, _ = interpret_semantic_intent(
                    question,
                    lambda _prompt, topic=topic: _payload(
                        task="list",
                        relation=None,
                        target_kind="none",
                        target_text=None,
                        filters=({
                            "field": "topic",
                            "operator": "related_to",
                            "value": topic,
                        },),
                        requested=("code", "name"),
                    ),
                )
                validation = validate_semantic_intent(intent, question)

                self.assertTrue(validation.valid, validation.reason)
                self.assertEqual((intent.task, intent.subject), ("list", "course"))
                self.assertEqual(intent.target.kind, "none")
                self.assertEqual(
                    [(item.field, item.operator, item.value) for item in intent.filters],
                    [("topic", "related_to", topic)],
                )

    def test_prompt_states_generic_course_attribute_discovery_boundary(self):
        prompt = build_semantic_interpreter_prompt("generic curriculum question")

        self.assertEqual(SEMANTIC_INTERPRETER_PROMPT_VERSION, "semantic-interpreter/v9")
        self.assertIn("LITERAL COURSE ATTRIBUTE LOOKUP vs TOPIC/COURSE DISCOVERY", prompt)
        self.assertIn("information about one apparent course X", prompt)
        self.assertIn("description of course X", prompt)
        self.assertIn("set or list of courses related to X", prompt)
        self.assertIn("not a topic filter", prompt)


if __name__ == "__main__":
    unittest.main()
