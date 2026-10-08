"""Provider-free G4 linguistic contract and omission controls."""

import copy
import json
import unittest

from rag.semantic.interpreter import (
    SemanticSchemaError, interpret_semantic_intent,
    parse_semantic_intent_payload, semantic_intent_json_schema,
)
from rag.semantic.schema import ComparisonSpec, SemanticFilter, SemanticIntent, SemanticTarget
from rag.semantic.validation import validate_semantic_intent


def payload(references=("06016481", "06016482"), fields=("placement", "alternative_selection")):
    return {
        "task": "lookup", "subject": "course", "relation": "placement",
        "target": {
            "kind": "literal_set", "raw_text": None,
            "normalized_hint": None, "ordinal": None,
            "members": [{"raw_text": text, "normalized_hint": None} for text in references],
        },
        "scope": {"program": None, "catalog": None, "plan": None,
                  "plan_hint": None, "year": None, "semester": None},
        "filters": [], "aggregation": None, "ranking": None, "comparison": None,
        "requested_fields": list(fields), "clarification": None,
        "policy_topic": None, "observed_value": None,
    }


def parse(data):
    return parse_semantic_intent_payload(json.dumps(data))


class CourseSetContractTests(unittest.TestCase):
    def test_two_codes_and_alternative_request(self):
        intent = parse(payload())
        self.assertEqual([m.raw_text for m in intent.target.members], ["06016481", "06016482"])
        self.assertIn("alternative_selection", intent.requested_fields)
        self.assertTrue(validate_semantic_intent(intent, "Compare availability of 06016481 and 06016482").valid)

    def test_three_codes_and_sequence_request(self):
        intent = parse(payload(("06016413", "06016420", "06016421"),
                               ("placement", "prerequisites", "placement_sequence")))
        self.assertEqual(len(intent.target.members), 3)
        self.assertIn("placement_sequence", intent.requested_fields)
        self.assertTrue(validate_semantic_intent(intent, "Arrange 06016413, 06016420, 06016421").valid)

    def test_independent_compound_references(self):
        refs = ("PROJECT 1 (06016406)", "SERVER SIDE WEB DEVELOPMENT (06016418)")
        intent = parse(payload(refs, ("placement",)))
        self.assertEqual(tuple(m.raw_text for m in intent.target.members), refs)
        self.assertTrue(validate_semantic_intent(intent, " and ".join(refs)).valid)

    def test_schema_has_closed_member_objects_and_request_vocabulary(self):
        schema = semantic_intent_json_schema()
        target = schema["properties"]["target"]
        self.assertIn("literal_set", target["properties"]["kind"]["enum"])
        members = target["properties"]["members"]
        self.assertEqual(members["maxItems"], 20)
        self.assertFalse(members["items"]["additionalProperties"])
        self.assertEqual(set(members["items"]["required"]), {"raw_text", "normalized_hint"})
        self.assertIn("alternative_selection", schema["properties"]["relation"]["anyOf"][0]["enum"])
        self.assertIn("placement_sequence", schema["properties"]["requested_fields"]["items"]["enum"])

    def test_one_argument_callable_once(self):
        calls = []
        def stub(prompt):
            calls.append(prompt)
            return json.dumps(payload())
        intent, _ = interpret_semantic_intent("When can 06016481 and 06016482 be taken?", stub)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(intent.target.members), 2)

    def test_legacy_singleton_payload_unchanged(self):
        data = payload(fields=("placement",))
        data["target"] = {"kind": "literal", "raw_text": "06016481",
                          "normalized_hint": None, "ordinal": None}
        self.assertEqual(parse(data).target.raw_text, "06016481")

    def test_invalid_member_shapes_fail_closed(self):
        base = payload()
        variants = []
        for members in ([], [base["target"]["members"][0]], ["06016481", "06016482"],
                        [None, None], base["target"]["members"] * 11):
            data = copy.deepcopy(base)
            data["target"]["members"] = members
            variants.append(data)
        for key, value in (("extra", 1), ("members", []), ("kind", "literal_set"),
                           ("minimum_choices", 1)):
            data = copy.deepcopy(base)
            data["target"]["members"][0][key] = value
            variants.append(data)
        data = copy.deepcopy(base)
        data["target"]["extra"] = 1
        variants.append(data)
        data = copy.deepcopy(base)
        data["target"]["raw_text"] = "combined target"
        variants.append(data)
        data = copy.deepcopy(base)
        data["target"]["raw_text"] = ""
        variants.append(data)
        for data in variants:
            with self.subTest(target=data["target"]), self.assertRaises(SemanticSchemaError):
                parse(data)

    def test_non_set_cannot_carry_members(self):
        data = payload()
        data["target"].update(kind="literal", raw_text="06016481")
        with self.assertRaises(SemanticSchemaError):
            parse(data)

    def test_code_omission_rejects_singleton(self):
        intent = SemanticIntent(task="lookup", relation="placement",
                                target=SemanticTarget(kind="literal", raw_text="06016481"))
        result = validate_semantic_intent(intent, "06016481 and 06016482 placement")
        self.assertFalse(result.valid)
        self.assertIn("06016482", result.reason)

    def test_codes_and_topic_words_cannot_turn_into_discovery(self):
        intent = SemanticIntent(task="list", filters=(SemanticFilter("topic", "related_to", "infrastructure"),))
        result = validate_semantic_intent(intent, "infrastructure 06016413 06016420 06016421")
        self.assertFalse(result.valid)

    def test_invented_member_rejected(self):
        self.assertFalse(validate_semantic_intent(parse(payload()), "only 06016481 is supplied").valid)

    def test_set_discovery_filter_rejected(self):
        data = payload()
        data["filters"] = [{"field": "topic", "operator": "related_to", "value": "networks"}]
        self.assertFalse(validate_semantic_intent(parse(data), "networks 06016481 06016482").valid)

    def test_combined_codes_are_not_one_member(self):
        intent = parse(payload(("06016481 and 06016482", "06016413")))
        self.assertFalse(validate_semantic_intent(intent, "06016481 and 06016482 plus 06016413").valid)

    def test_invalid_task_subject_stays_rejected(self):
        data = payload()
        data["subject"] = "requirement"
        self.assertFalse(validate_semantic_intent(parse(data), "06016481 06016482").valid)

    def test_topic_discovery_unchanged(self):
        intent = SemanticIntent(task="list", filters=(SemanticFilter("topic", "related_to", "marine robotics"),))
        self.assertTrue(validate_semantic_intent(intent, "courses related to marine robotics").valid)

    def test_unknown_fields_stay_rejected(self):
        data = payload()
        data["invented"] = None
        with self.assertRaises(SemanticSchemaError):
            parse(data)

    def test_selection_only_relation_is_course_lookup(self):
        data = payload(fields=("alternative_selection",))
        data["relation"] = "alternative_selection"
        intent = parse(data)
        self.assertTrue(validate_semantic_intent(intent, "Choose from 06016481 and 06016482").valid)

    def test_code_boundaries_do_not_ground_substrings(self):
        intent = SemanticIntent(task="lookup", relation="placement",
                                target=SemanticTarget(kind="literal", raw_text="06016413"))
        self.assertFalse(validate_semantic_intent(intent, "106016413 placement").valid)

    def test_comparison_code_coverage_preserves_operand_isolation(self):
        intent = SemanticIntent(task="compare", comparison=ComparisonSpec(
            left=(("course", "06016413"),), right=(("course", "06016420"),),
            measure="credits", operation="difference"))
        self.assertTrue(validate_semantic_intent(intent, "06016413 vs 06016420 credits").valid)
        self.assertFalse(validate_semantic_intent(intent, "06016413 vs 06016420 and 06016421 credits").valid)

    def test_schema_conditional_set_and_legacy_shapes(self):
        variants = semantic_intent_json_schema()["properties"]["target"]["anyOf"]
        self.assertEqual(variants[0]["required"], ["members"])
        self.assertEqual(variants[0]["properties"]["members"]["minItems"], 2)
        self.assertEqual(variants[1]["properties"]["members"]["maxItems"], 0)

    def test_member_first_mention_order_is_enforced(self):
        self.assertFalse(validate_semantic_intent(parse(payload()), "06016482 then 06016481").valid)


if __name__ == "__main__":
    unittest.main()
