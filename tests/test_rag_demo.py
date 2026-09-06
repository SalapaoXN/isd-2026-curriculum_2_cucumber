import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.demo import run_demo


class RagDemoTest(unittest.TestCase):
    def test_rebuilds_existing_database_and_sidecars_before_each_run(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text("{}", encoding="utf-8")
            database_path.write_text("old database", encoding="utf-8")
            sidecars = [
                Path(f"{database_path}-wal"),
                Path(f"{database_path}-shm"),
                Path(f"{database_path}-journal"),
            ]
            for sidecar in sidecars:
                sidecar.write_text("old sidecar", encoding="utf-8")

            loader_calls = []

            def fake_loader(source_path, output_path):
                loader_calls.append((source_path, output_path))
                self.assertFalse(output_path.exists())
                self.assertFalse(any(sidecar.exists() for sidecar in sidecars))
                output_path.write_text("new database", encoding="utf-8")

            with patch("rag.demo.load_json_to_sqlite", side_effect=fake_loader), patch(
                "rag.demo.build_chunks", return_value=[{"text": "chunk"}]
            ), patch("rag.demo.embed_texts", return_value=object()), patch(
                "rag.demo.insert_embeddings"
            ), patch("rag.demo.retrieve", return_value=[]):
                run_demo(input_path, database_path, "วิชา", top_k=1)
                run_demo(input_path, database_path, "วิชา", top_k=1)

            self.assertEqual(len(loader_calls), 2)


if __name__ == "__main__":
    unittest.main()
