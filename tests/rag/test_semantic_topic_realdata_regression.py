"""Provider-free real-data regressions for semantic topic discovery."""

import json
from pathlib import Path
import unittest

from rag.semantic.modes import semantic_ask_response
from rag.retrieval.retrieve import retrieve_constrained_topic_evidence
from rag.structured.queries import scoped_course_set
from tests.rag.test_semantic_clarification_transport import payload


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"


def topic_payload(topic):
    data = payload(None)
    data.update(task="list", subject="course", relation=None, comparison=None)
    data["target"] = {
        "kind": "none", "raw_text": None, "normalized_hint": None,
        "ordinal": None, "members": [],
    }
    data["scope"].update(program=None, catalog=None, plan=None, plan_hint=None,
                         year=None, semester=None)
    data["filters"] = [{"field": "topic", "operator": "related_to", "value": topic}]
    data["aggregation"] = None
    data["ranking"] = None
    data["requested_fields"] = ["code", "name"]
    return json.dumps(data, ensure_ascii=False)


class SemanticTopicRealDataRegressionTests(unittest.TestCase):
    def run_topic(self, question, topic, context):
        return semantic_ask_response(
            DB, question, context, home_program=context.get("program"),
            interpret_callable=lambda prompt: topic_payload(topic),
        )

    def test_diagnostic_english_and_calculation_topics(self):
        contexts = (
            {"program": "DSBA", "catalog_key": "dsba-2565", "plan": "coop"},
            {"program": "DSBA", "catalog_key": "dsba-2565", "plan": "no_coop"},
            {"program": "IT", "catalog_key": "it-2565", "plan": "coop"},
        )
        cases = (
            ("วิชาที่เกี่ยวข้องกับภาษาอังกฤษมีอะไรบ้าง", "ภาษาอังกฤษ"),
            ("วิชาที่เกี่ยวข้องกับการคำนวณมีอะไรบ้าง", "การคำนวณ"),
        )
        for context in contexts:
            for question, topic in cases:
                with self.subTest(context=context, topic=topic):
                    response = self.run_topic(question, topic, context)
                    print("TOPIC_DIAG", context, topic, response.get("status"), response.get("answer"))
                    self.assertIn(response["status"], {"answer", "insufficient_evidence"})

    def test_diagnostic_retrieval_distances(self):
        for program, catalog, plan in (
            ("DSBA", "dsba-2565", "coop"),
            ("IT", "it-2565", "coop"),
        ):
            structural = scoped_course_set(DB, program, (plan,), catalog_key=catalog)
            self.assertEqual(structural["status"], "ok")
            candidates = structural["courses"]
            for topic in ("ภาษาอังกฤษ", "การคำนวณ"):
                with self.subTest(program=program, topic=topic):
                    result = retrieve_constrained_topic_evidence(DB, topic, candidates)
                    distances = []
                    for candidate in result.candidates:
                        values = [
                            evidence.get("distance")
                            for evidence in candidate.get("description_evidence", ())
                            if evidence.get("distance") is not None
                        ]
                        if values:
                            distances.append((min(values), candidate.get("course_code"), candidate.get("name_th")))
                    distances.sort()
                    print("RETRIEVAL_DIAG", program, topic, result.status,
                          "candidates", len(result.candidates),
                          "matches", len(result.scored_candidates),
                          "missing_desc", len(result.missing_description_course_ids),
                          "best", distances[:8], "worst", distances[-3:])
                    self.assertTrue(result.status)


if __name__ == "__main__":
    unittest.main()
