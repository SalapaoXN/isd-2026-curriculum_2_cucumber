"""Regressions for opt-in local process logging."""

from contextlib import redirect_stdout
from io import StringIO
import os
import unittest
from unittest.mock import patch

from rag.process_log import PROCESS_LOG_PREFIX, process_log


class ProcessLogTests(unittest.TestCase):
    def test_disabled_by_default_is_silent(self):
        output = StringIO()
        with patch.dict(os.environ, {"CUCUMBER_PROCESS_LOG": ""}, clear=False):
            with redirect_stdout(output):
                process_log("request", question="ทดสอบ")
        self.assertEqual(output.getvalue(), "")

    def test_enabled_prints_bounded_json_record(self):
        output = StringIO()
        with patch.dict(os.environ, {"CUCUMBER_PROCESS_LOG": "1"}, clear=False):
            with redirect_stdout(output):
                process_log("request", question="วิชาที่เกี่ยวข้องกับการคำนวณ")
        rendered = output.getvalue()
        self.assertIn(PROCESS_LOG_PREFIX, rendered)
        self.assertIn('"stage":"request"', rendered)
        self.assertIn("การคำนวณ", rendered)


if __name__ == "__main__":
    unittest.main()
