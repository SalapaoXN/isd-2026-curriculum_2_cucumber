"""P1.3 static runbook checks: semantic startup is unambiguous and verifiable."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.semantic.modes import (
    DEFAULT_QA_MODE,
    QA_MODE_ENV_VAR,
    QA_MODE_SEMANTIC,
    active_qa_mode,
)


ROOT = Path(__file__).resolve().parents[2]


class SemanticStartupRunbookTests(unittest.TestCase):
    def test_semantic_mode_activates_only_with_explicit_env(self):
        with patch.dict(os.environ, {QA_MODE_ENV_VAR: "semantic"}):
            self.assertEqual(active_qa_mode(), QA_MODE_SEMANTIC)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(QA_MODE_ENV_VAR, None)
            self.assertEqual(active_qa_mode(), DEFAULT_QA_MODE)
            self.assertNotEqual(active_qa_mode(), QA_MODE_SEMANTIC)

    def test_readme_documents_semantic_launch_and_health_contract(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("$env:CUCUMBER_QA_MODE = \"semantic\"", readme)
        self.assertIn("uvicorn backend.main:app", readme)
        self.assertIn("qa_mode = semantic", readme)
        self.assertIn("database_ready = true", readme)
        self.assertIn("status = ok", readme)
        self.assertIn("run_semantic.ps1", readme)

    def test_run_semantic_script_sets_mode_and_checks_health(self):
        script = (ROOT / "scripts" / "run_semantic.ps1").read_text(encoding="utf-8")
        self.assertIn('$env:CUCUMBER_QA_MODE = "semantic"', script)
        self.assertIn("uvicorn", script)
        self.assertIn("backend.main:app", script)
        self.assertIn("/api/health", script)
        self.assertIn("qa_mode", script)


if __name__ == "__main__":
    unittest.main()
