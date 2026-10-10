import tempfile
import unittest
from unittest import mock
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main, rag_service
from backend.schemas import AnswerDraft, GeneralDraft, RouteDecision


class RAGServiceTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close()

    def test_general_conversation_skips_retrieval(self):
        def generate(_prompt, _payload, schema):
            return GeneralDraft(answer="Hello! How can I help with your curriculum?")

        with patch.object(rag_service, "_classify_use_rag", return_value=False), \
             patch.object(rag_service, "_generate", side_effect=generate), \
             patch.object(rag_service, "_search", side_effect=AssertionError("retrieval should be skipped")):
            result = rag_service.answer("Hello", [])
        self.assertEqual(result["status"], "general")
        self.assertEqual(result["sources"], [])
        self.assertFalse(result["database_searched"])

    def test_success_logs_route_and_result_without_question_text(self):
        secret_question = "private question text 0600"
        with patch.object(rag_service, "_classify_use_rag", return_value=False), \
             patch.object(rag_service, "_generate", return_value=GeneralDraft(answer="Hello")), \
             self.assertLogs(rag_service._logger, level="INFO") as captured:
            result = rag_service.answer(secret_question, [])
        log_text = "\n".join(captured.output)
        self.assertNotIn(secret_question, log_text)
        self.assertIn("event=classified use_rag=False", log_text)
        self.assertIn("database_searched=False", log_text)
        self.assertIn(result["request_id"], log_text)

    def test_ambiguous_request_returns_clarification_without_search(self):
        decision = RouteDecision(intent="clarification", use_rag=False,
                                 standalone_query="Which course?", needs_clarification=True,
                                 clarification_question="Which program and curriculum year do you mean?")
        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", return_value=decision), \
             patch.object(rag_service, "_search", side_effect=AssertionError("retrieval should be skipped")):
            result = rag_service.answer("What does that require?", [])
        self.assertEqual(result["status"], "clarification_required")
        self.assertIn("Which program", result["answer"])
        self.assertFalse(result["database_searched"])

    def test_course_answer_citations_are_limited_to_retrieved_chunks(self):
        decision = RouteDecision(intent="course_question", use_rag=True,
                                 standalone_query="IT year one courses", needs_clarification=False,
                                 clarification_question=None)
        chunks = [
            {"chunk_id": "real-chunk", "text": "Year 1 includes course A", "program": "IT", "distance": 0.4,
             "plan": "coop", "course_code": "ABC100", "provenance": [{"source_page": 9,
             "source_filename": "it-plan.pdf"}]},
            {"chunk_id": "unselected-chunk", "text": "Other course information", "program": "IT", "distance": 0.5},
        ]

        def generate(_prompt, _payload, schema):
            if schema is RouteDecision:
                return decision
            return AnswerDraft(sufficient=True, answer="Year 1 includes course A.", source_ids=["real-chunk"])

        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", side_effect=generate), \
             patch.object(rag_service, "_search", return_value=chunks):
            result = rag_service.answer("What courses are in IT year one?", [])
        self.assertEqual(result["status"], "answer")
        self.assertTrue(result["database_searched"])
        self.assertEqual([source["chunk_id"] for source in result["sources"]], ["real-chunk"])
        self.assertEqual(result["sources"][0]["source_page"], 9)

    def test_exhaustive_course_list_uses_structured_database_and_formats_every_item(self):
        decision = RouteDecision(
            intent="course_question", use_rag=True, standalone_query="IT 2565 year 1 semester 1",
            needs_clarification=False, clarification_question=None, exhaustive_course_list=True,
            program_code="IT", academic_year="2565", study_year=1, semester=1,
        )
        courses = [{
            "course_id": index, "course_code": code, "name_th": None, "name_en": name,
            "credits": "3(3-0-6)", "academic_year": "2565", "program_code": "IT",
            "plans": ["coop", "no_coop"], "sources": [f"it2565_page_{index}.png · p.{index}"],
        } for index, code, name in ((1, "IT101", "Course One"), (2, "IT102", "Course Two"))]

        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", return_value=decision), \
             patch("backend.curriculum.list_courses", return_value={"items": courses, "total": 2}), \
             patch.object(rag_service, "_search", side_effect=AssertionError("course lists use structured retrieval")):
            result = rag_service.answer("List all IT 2565 year 1 semester 1 courses", [])

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["retrieval_count"], 2)
        self.assertIn("2 subjects", result["answer"])
        self.assertIn("1. IT101 — Course One — 3(3-0-6) credits", result["answer"])
        self.assertIn("2. IT102 — Course Two — 3(3-0-6) credits", result["answer"])
        self.assertEqual(len(result["sources"]), 2)

    def test_top_k_includes_all_chunks_tied_at_the_boundary(self):
        distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6, 0.7]
        calls = []

        def fetch(count):
            calls.append(count)
            return [{"chunk_id": str(index), "distance": distance}
                    for index, distance in enumerate(distances[:count])]

        hits = rag_service._top_k_with_ties(fetch, limit=6)
        self.assertEqual(len(hits), 8)
        self.assertEqual(calls, [7, 14])

    def test_unsupported_answer_is_returned_without_citations(self):
        decision = RouteDecision(intent="course_question", use_rag=True,
                                 standalone_query="Scholarships for IT", needs_clarification=False,
                                 clarification_question=None)
        def generate(_prompt, _payload, schema):
            if schema is RouteDecision:
                return decision
            return AnswerDraft(sufficient=False, answer="The indexed sources do not provide scholarship details.", source_ids=[])

        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", side_effect=generate), \
             patch.object(rag_service, "_search", return_value=[{"chunk_id": "curriculum", "text": "course list", "distance": 0.2}]):
            result = rag_service.answer("What scholarships are available?", [])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["sources"], [])
        self.assertTrue(result["database_searched"])

    def test_invalid_classifier_output_fails_safe(self):
        with patch.object(rag_service, "_classify_use_rag", side_effect=ValueError("invalid output")), \
             patch.object(rag_service, "_search", side_effect=AssertionError("must not retrieve")):
            with self.assertRaises(Exception) as error:
                rag_service.answer("What is the credit requirement?", [])
        self.assertEqual(error.exception.status_code, 503)

    def test_mini_classifier_returns_only_a_valid_boolean(self):
        provider = mock.Mock(return_value="true\n")
        with patch.object(rag_service, "_classifier_provider", return_value=provider):
            result = rag_service._classify_use_rag("How many credits?", [])
        self.assertIs(result, True)
        args, kwargs = provider.call_args
        self.assertEqual(kwargs["response_mime_type"], "text/plain")
        self.assertIn('"latest_message": "How many credits?"', args[0])

    def test_classifier_uses_a_distinct_mini_model(self):
        rag_service._classifier_provider.cache_clear()
        fake_provider = mock.Mock()
        with patch.object(rag_service, "make_gemini_callable", return_value=fake_provider) as factory:
            self.assertIs(rag_service._classifier_provider(), fake_provider)
        factory.assert_called_once_with("gemini-3.1-flash-lite")
        rag_service._classifier_provider.cache_clear()

    def test_mini_classifier_rejects_non_boolean_output(self):
        for value in ("probably true", '{"use_rag": true}', "", "1"):
            with self.subTest(value=value):
                provider = mock.Mock(return_value=value)
                with patch.object(rag_service, "_classifier_provider", return_value=provider):
                    with self.assertRaises(ValueError):
                        rag_service._classify_use_rag("Hello", [])

    def test_missing_index_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(rag_service, "INDEX_PATH", Path(directory) / "missing.db"):
            with self.assertRaises(Exception) as error:
                rag_service._search("course query")
        self.assertEqual(error.exception.status_code, 503)
        self.assertIn("rag.build_index", str(error.exception.detail))

    def test_health_and_chat_route_contract(self):
        with patch("backend.main.health", return_value={"status": "ok", "database_ready": True}), \
             patch("backend.main.answer", return_value={"request_id": "req123", "status": "general", "answer": "Hello", "sources": []}):
            health = self.client.get("/api/health")
            answer = self.client.post("/api/chat", json={"question": "Hello", "history": []})
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(answer.json()["answer"], "Hello")
        self.assertEqual(answer.status_code, 200)
        self.assertEqual(answer.headers["X-Request-ID"], "req123")
        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        self.assertIn("Cucumber", root.text)

    def test_chat_request_rejects_whitespace_only_question(self):
        response = self.client.post("/api/chat", json={"question": "   ", "history": []})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
