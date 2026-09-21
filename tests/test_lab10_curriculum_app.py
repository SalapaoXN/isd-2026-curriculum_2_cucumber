import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from lab10_fastapi.curriculum_app import main


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
        response = self.client.post(
            "/api/ask",
            json={"question": "IT วิชา 06016454 มีกี่หน่วยกิต"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "answer")
        self.assertIn("3", payload["answer"])
        self.assertTrue(payload["provenance"])
        self.assertNotIn("sql", payload)
        self.assertNotIn("rows", payload)

    def test_provider_required_path_returns_controlled_fail_closed_status(self):
        response = self.client.post(
            "/api/ask",
            json={"question": "DSBA มีรายวิชาทั้งหมดเท่าไหร่"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "insufficient_evidence")

    def test_unsupported_request_remains_fail_closed(self):
        response = self.client.post(
            "/api/ask",
            json={"question": "IT 06016454 เรียนยากไหม"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unsupported")


if __name__ == "__main__":
    unittest.main()
