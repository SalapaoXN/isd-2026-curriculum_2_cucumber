import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import main


DB_PATH = Path(__file__).parents[1] / "cucumber_outputs" / "runtime" / "curriculum.db"


class CurriculumAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.db_patch = patch.object(main, "DEFAULT_CURRICULUM_DB_PATH", DB_PATH)
        self.db_patch.start()
        self.provider_patch = patch.object(
            main,
            "make_gemini_callable",
            side_effect=AssertionError("provider must remain lazy"),
        )
        self.provider_patch.start()
        main._provider = None

    def tearDown(self):
        main._provider = None
        self.provider_patch.stop()
        self.db_patch.stop()

    def test_import_does_not_create_provider(self):
        self.assertIsNone(main._provider)

    def test_health_works_without_provider(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["database_ready"])

    def test_missing_static_index_is_controlled(self):
        missing = Path(self.id()).with_name("missing-static-index.html")
        with patch.object(main, "STATIC_DIR", missing):
            with self.assertRaises(HTTPException) as error:
                main.index()

        self.assertEqual(error.exception.status_code, 404)

    def test_deterministic_ask_works_without_provider_and_has_truthful_schema(self):
        with patch.object(
            main,
            "ask_sql",
            return_value={
                "status": "answer",
                "answer": "พบข้อมูลรายวิชา",
                "sql": "SELECT ...",
                "columns": ["course_code"],
                "rows": [{"course_code": "06016454"}],
            },
        ) as sql_service:
            response = self.client.post(
                "/api/ask",
                json={"question": "IT วิชา 06016454 มีกี่หน่วยกิต"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "answer")
        self.assertEqual(payload["answer"], "พบข้อมูลรายวิชา")
        sql_service.assert_called_once()
        self.assertNotIn("sql", payload)
        self.assertNotIn("rows", payload)

    def test_ask_accepts_and_returns_bounded_single_course_context(self):
        focus = {
            "course_code": "06026201",
            "course_name": "CALCULUS 2",
            "program": "DSBA",
        }
        with patch.object(
            main,
            "ask_sql",
            return_value={
                "status": "answer",
                "answer": "CALCULUS 2 เรียนปี 1",
                "sql": "SELECT ...",
                "columns": ["course_code", "name_en"],
                "rows": [{"course_code": "06026201", "name_en": "CALCULUS 2"}],
                "next_context": {"program": "DSBA", "focus_course": focus},
            },
        ) as sql_service:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "แล้วเรียนปีไหน",
                    "conversation_context": {
                        "program": "DSBA",
                        "focus_course": focus,
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "answer")
        self.assertEqual(payload["next_context"]["program"], "DSBA")
        self.assertEqual(payload["next_context"]["focus_course"], focus)
        sql_service.assert_called_once()
        self.assertEqual(sql_service.call_args.args[1:3], ("แล้วเรียนปีไหน", "DSBA"))
        self.assertEqual(
            sql_service.call_args.kwargs["conversation_context"],
            {"program": "DSBA", "focus_course": focus},
        )
        self.assertNotIn("sql", payload)
        self.assertNotIn("rows", payload)

    def test_existing_program_only_context_remains_scoped(self):
        with patch.object(
            main,
            "ask_sql",
            return_value={"status": "answer", "answer": "พบข้อมูล"},
        ) as sql_service:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "ปี 3 เทอม 1 มีวิชาอะไรบ้าง",
                    "conversation_context": {"program": "IT"},
                },
            )

        self.assertEqual(response.status_code, 200)
        sql_service.assert_called_once()
        self.assertEqual(sql_service.call_args.args[2], "IT")
        self.assertEqual(
            sql_service.call_args.kwargs["conversation_context"],
            {"program": "IT"},
        )
        self.assertEqual(response.json()["next_context"], {"program": "IT"})

    def test_ask_chains_bounded_result_courses_through_context_contract(self):
        courses = [
            {"program": "IT", "course_code": "06016401", "course_name": "Course A"},
            {"program": "IT", "course_code": "06016402", "course_name": "Course B"},
        ]
        context = {
            "program": "IT",
            "result_courses": courses,
            "result_scope_program": "IT",
        }
        with patch.object(
            main,
            "ask_sql",
            return_value={
                "status": "answer",
                "answer": "พบวิชาที่มี prerequisite",
                "next_context": context,
            },
        ) as sql_service:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "ตัวไหนมีวิชาบังคับก่อน",
                    "conversation_context": context,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["next_context"], context)
        self.assertEqual(sql_service.call_args.args[2], "IT")
        self.assertEqual(
            sql_service.call_args.kwargs["conversation_context"],
            {
                "program": "IT",
                "result_courses": courses,
                "result_scope_program": "IT",
            },
        )
        self.assertNotIn("rows", response.json())
        self.assertNotIn("sql", response.json())

    def test_ask_chains_known_empty_result_set_context(self):
        context = {
            "program": "AIT",
            "result_courses": [],
            "result_set_empty": True,
            "result_scope_program": "AIT",
        }
        with patch.object(
            main,
            "ask_sql",
            return_value={
                "status": "no_data",
                "answer": "จากรายการก่อนหน้า ไม่พบรายการที่ตรงกับเงื่อนไขนี้",
                "next_context": context,
            },
        ) as sql_service:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "ในวิชาเหล่านี้รวมกี่หน่วยกิต",
                    "conversation_context": context,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["next_context"], context)
        self.assertEqual(
            sql_service.call_args.kwargs["conversation_context"],
            {
                "program": "AIT",
                "result_courses": [],
                "result_set_empty": True,
                "result_scope_program": "AIT",
            },
        )

    def test_provider_required_path_returns_controlled_fail_closed_status(self):
        def unavailable_service(*args, **kwargs):
            with self.assertRaises(main.ProviderUnavailable):
                args[3]("prompt")
            return {"status": "error", "answer": "", "error": {"code": "sql_model_failure"}}

        with patch.object(main, "ask_sql", side_effect=unavailable_service):
            response = self.client.post(
                "/api/ask",
                json={"question": "DSBA มีรายวิชาทั้งหมดเท่าไหร่"},
            )

        self.assertEqual(response.status_code, 503)

    def test_unsupported_request_remains_fail_closed(self):
        with patch.object(
            main,
            "ask_sql",
            return_value={
                "status": "error",
                "answer": "",
                "error": {"code": "invalid_sql"},
            },
        ):
            response = self.client.post(
                "/api/ask",
                json={"question": "IT 06016454 เรียนยากไหม"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "error")
        self.assertEqual(response.json()["action"], "invalid_sql")


if __name__ == "__main__":
    unittest.main()
