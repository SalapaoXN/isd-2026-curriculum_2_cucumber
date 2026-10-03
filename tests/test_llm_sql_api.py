import unittest
import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main
from backend.hard_qa import HARD_INTERPRETATION_RESPONSE_JSON_SCHEMA, answer_hard_question


DB_PATH = Path(__file__).parents[1] / "cucumber_outputs" / "runtime" / "curriculum.db"


class LlmSqlApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        main._provider = None
        self.db_patch = patch.object(main, "DEFAULT_CURRICULUM_DB_PATH", DB_PATH)
        self.db_patch.start()
        self._sql_service_patch = patch.object(main, "ask_sql", create=True)
        self.sql_service = self._sql_service_patch.start()
        self.sql_service.return_value = {
            "status": "answer",
            "answer": "พบ 4 วิชา",
            "sql": "SELECT ...",
            "columns": ["course_code"],
            "rows": [{"course_code": "C101"}],
        }

    def tearDown(self):
        self._sql_service_patch.stop()
        self.db_patch.stop()
        main._provider = None

    def test_api_ask_uses_sql_service_with_question_and_selected_program(self):
        with patch("rag.hybrid_demo.answer_question_once") as old_rag:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "ปี 3 เทอม 1 มีวิชาอะไรบ้าง",
                    "conversation_context": {"program": "IT"},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "พบ 4 วิชา")
        self.assertEqual(response.json()["status"], "answer")
        self.assertEqual(response.json()["route"], "llm_sql")
        self.assertEqual(response.json()["next_context"], {"program": "IT"})
        self.assertNotIn("sql", response.json())
        self.assertNotIn("rows", response.json())
        self.assertEqual(self.sql_service.call_args.args[:3], (
            DB_PATH,
            "ปี 3 เทอม 1 มีวิชาอะไรบ้าง",
            "IT",
        ))
        self.assertTrue(callable(self.sql_service.call_args.args[3]))
        self.assertTrue(callable(self.sql_service.call_args.args[4]))
        old_rag.assert_not_called()

    def test_catalog_key_survives_api_result_context_round_trip(self):
        edition_context = {
            "program": "DSBA",
            "result_courses": [
                {"catalog_key": "dsba-2560", "program": "DSBA", "course_code": "C101"}
            ],
            "result_scope_program": "DSBA",
        }
        self.sql_service.return_value = {
            "status": "answer",
            "answer": "พบ 1 วิชา",
            "next_context": edition_context,
        }

        initial = self.client.post(
            "/api/ask",
            json={
                "question": "ปี 3 เทอม 1 มีวิชาอะไรบ้าง",
                "conversation_context": {"program": "DSBA"},
            },
        )

        self.assertEqual(initial.status_code, 200)
        self.assertEqual(initial.json()["next_context"], edition_context)

        follow_up = self.client.post(
            "/api/ask",
            json={
                "question": "ในวิชาเหล่านี้มีกี่วิชา",
                "conversation_context": initial.json()["next_context"],
            },
        )

        self.assertEqual(follow_up.status_code, 200)
        forwarded = self.sql_service.call_args.kwargs["conversation_context"]
        self.assertEqual(
            forwarded["result_courses"][0]["catalog_key"], "dsba-2560"
        )

    def test_question_without_program_passes_none_and_returns_service_status(self):
        self.sql_service.return_value = {
            "status": "no_data",
            "answer": "ไม่พบข้อมูลที่ตรงกับคำถาม",
            "sql": "SELECT ...",
            "columns": ["course_code"],
            "rows": [],
        }

        response = self.client.post("/api/ask", json={"question": "รหัสวิชา X999 คืออะไร"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.sql_service.call_args.args[1:3], ("รหัสวิชา X999 คืออะไร", None))
        self.assertEqual(response.json()["status"], "no_data")
        self.assertEqual(response.json()["next_context"], None)

    def test_controlled_service_failure_maps_to_existing_response_contract(self):
        self.sql_service.return_value = {
            "status": "error",
            "answer": "",
            "sql": "",
            "columns": [],
            "rows": [],
            "error": {"code": "invalid_sql"},
        }

        response = self.client.post("/api/ask", json={"question": "แสดงวิชา"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "error")
        self.assertEqual(response.json()["action"], "invalid_sql")
        self.assertEqual(response.json()["answer"], "")

    def test_provider_unavailability_remains_http_503(self):
        self.sql_service.side_effect = main.ProviderUnavailable("provider unavailable")
        response = self.client.post("/api/ask", json={"question": "แสดงวิชา"})

        self.assertEqual(response.status_code, 503)

    def test_programs_endpoint_remains_available(self):
        response = self.client.get("/api/programs")

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json()["programs"], list)

    def test_curriculum_endpoint_remains_available(self):
        response = self.client.get("/api/curriculum", params={"program": "IT", "limit": 1})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["limit"], 1)
        self.assertIsInstance(response.json()["items"], list)

    def test_hard_comparison_uses_deterministic_h1_and_returns_provenance(self):
        question = "DSBA coop กับ no_coop ต่างกันที่วิชาไหน"
        context = {"program": "DSBA"}
        intent = {
            "task_type": "plan_comparison",
            "program": "DSBA",
            "plan": None,
            "left_plan": "coop",
            "right_plan": "no_coop",
            "target_course_code": None,
            "horizon_terms": None,
        }
        expected = answer_hard_question(DB_PATH, question, context, lambda prompt: json.dumps(intent))
        with patch.object(main, "_lazy_provider", return_value=json.dumps(intent)) as provider:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": question,
                    "conversation_context": context,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["route"], "hard")
        self.assertEqual(payload["hard_task_type"], "plan_comparison")
        self.assertEqual(payload["status"], "answer")
        self.assertIn("ไม่พบความแตกต่างของชุดรหัสวิชา", payload["answer"])
        self.assertTrue(payload["provenance"])
        self.assertIn("อ้างอิง:", payload["answer"])
        self.assertEqual(payload["provenance"], expected["provenance"])
        self.assertTrue(all("source_filename" in item and "document_page" in item for item in payload["provenance"]))
        self.assertEqual(self.sql_service.call_count, 0)
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(provider.call_args.kwargs["response_mime_type"], "application/json")
        self.assertEqual(provider.call_args.kwargs["response_json_schema"], HARD_INTERPRETATION_RESPONSE_JSON_SCHEMA)
        self.assertNotIn("response_schema", provider.call_args.kwargs)

    def test_rejected_hard_interpretation_does_not_fall_through_to_sql_service(self):
        with patch.object(main, "_lazy_provider", return_value="not JSON") as provider:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "DSBA แผน coop กับ no_coop ต่างกันที่วิชาไหนบ้าง",
                    "conversation_context": {"program": "DSBA"},
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["route"], "hard")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["action"], "hard_interpretation_failure")
        self.assertIn("ยังจัดประเภทคำถาม Hard นี้ไม่ได้อย่างปลอดภัย", payload["answer"])
        provider.assert_called_once()
        self.assertEqual(provider.call_args.kwargs["response_mime_type"], "application/json")
        self.assertEqual(provider.call_args.kwargs["response_json_schema"], HARD_INTERPRETATION_RESPONSE_JSON_SCHEMA)
        self.assertNotIn("response_schema", provider.call_args.kwargs)
        self.sql_service.assert_not_called()


if __name__ == "__main__":
    unittest.main()
