import unittest
import tempfile
from pathlib import Path

from src.pipeline.tools.runtime_artifacts import (
    audit_shared_course_conflicts,
    canonicalize_runtime_artifacts,
)


class FinalArtifactConflictPreflightTests(unittest.TestCase):
    def test_shared_course_facts_have_no_unresolved_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            canonical_dir = Path(directory) / "canonical"
            result = canonicalize_runtime_artifacts(output_dir=canonical_dir)
            conflicts = audit_shared_course_conflicts(result.paths)

        self.assertFalse(
            conflicts,
            "Unresolved canonical shared-course conflicts:\n"
            + "\n".join(conflicts),
        )
        self.assertEqual(len(result.paths), 15)


if __name__ == "__main__":
    unittest.main()
