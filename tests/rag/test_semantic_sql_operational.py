"""P1.1 SQL-bridge operational failures stay operational, never evidence gaps.

``execute_sql_bridge`` must surface SQL/answer provider failures as
``SemanticOperationalError("provider_unavailable")`` instead of
collapsing them into ``missing_data``. Non-provider SQL failures keep
their existing ``missing_data`` behavior.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

from rag.semantic.errors import SemanticOperationalError
from rag.semantic.executor import _missing_answer_provider, execute_sql_bridge
from rag.semantic.schema import (
    ResolvedIntent,
    ResolvedScope,
    SemanticIntent,
)


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"


def _list_resolved():
    return ResolvedIntent(
        intent=SemanticIntent(task="list", subject="course"),
        scope=ResolvedScope(program="IT", catalog_key="it-2565"),
    )


def _service_context():
    return {"program": "IT", "catalog_key": "it-2565"}


class SqlBridgeOperationalTests(unittest.TestCase):
    def test_sql_provider_timeout_is_operational(self):
        def _timeout(prompt, **kwargs):
            raise TimeoutError("request timed out")

        with self.assertRaises(SemanticOperationalError) as raised:
            execute_sql_bridge(
                DB, _list_resolved(), None, "list",
                _timeout, lambda *args, **kwargs: "",
                _service_context(),
            )
        self.assertEqual(raised.exception.status, "provider_unavailable")

    def test_sql_model_failure_dict_is_operational(self):
        with patch(
            "backend.llm_sql_qa.ask_sql",
            return_value={"status": "error", "answer": "",
                          "error": {"code": "sql_model_failure"}},
        ):
            with self.assertRaises(SemanticOperationalError) as raised:
                execute_sql_bridge(
                    DB, _list_resolved(), None, "list",
                    lambda *args, **kwargs: "SELECT 1",
                    lambda *args, **kwargs: "answer",
                    _service_context(),
                )
        self.assertEqual(raised.exception.status, "provider_unavailable")

    def test_answer_model_failure_with_real_provider_is_operational(self):
        with patch(
            "backend.llm_sql_qa.ask_sql",
            return_value={"status": "error", "answer": "",
                          "error": {"code": "answer_model_failure"}},
        ):
            with self.assertRaises(SemanticOperationalError) as raised:
                execute_sql_bridge(
                    DB, _list_resolved(), None, "list",
                    lambda *args, **kwargs: "SELECT 1",
                    lambda *args, **kwargs: "answer",
                    _service_context(),
                )
        self.assertEqual(raised.exception.status, "provider_unavailable")

    def test_answer_model_failure_without_provider_stays_missing_data(self):
        with patch(
            "backend.llm_sql_qa.ask_sql",
            return_value={"status": "error", "answer": "",
                          "error": {"code": "answer_model_failure"}},
        ):
            verified = execute_sql_bridge(
                DB, _list_resolved(), None, "list",
                lambda *args, **kwargs: "SELECT 1",
                _missing_answer_provider,
                _service_context(),
            )
        self.assertEqual(verified.status, "missing_data")

    def test_non_provider_sql_failure_stays_missing_data(self):
        with patch(
            "backend.llm_sql_qa.ask_sql",
            return_value={"status": "error", "answer": "",
                          "error": {"code": "invalid_sql"}},
        ):
            verified = execute_sql_bridge(
                DB, _list_resolved(), None, "list",
                lambda *args, **kwargs: "SELECT 1",
                lambda *args, **kwargs: "answer",
                _service_context(),
            )
        self.assertEqual(verified.status, "missing_data")


if __name__ == "__main__":
    unittest.main()
