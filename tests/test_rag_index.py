import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from rag.retrieval.index import ensure_index, query_index


class RagIndexTest(unittest.TestCase):
    def _write_source(self, path: Path, value: str = "first") -> None:
        path.write_text(
            json.dumps(
                {
                    "program": "TEST",
                    "plan": "regular",
                    "value": value,
                    "courses": [{"code": "C100", "name_th": value}],
                }
            ),
            encoding="utf-8",
        )

    def test_builds_once_and_persists_metadata_and_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "curriculum.json"
            artifact_dir = directory_path / "rag_artifacts"
            self._write_source(source_path)
            chunks = [
                {
                    "chunk_id": "chunk-1",
                    "text": "C100 first",
                    "provenance": [{"source_page": 7}],
                }
            ]
            embed_calls = []

            def fake_embed(texts):
                values = list(texts)
                embed_calls.append(values)
                return [[0.0] * 384 for _ in values]

            def fake_loader(_source, output):
                Path(output).touch()

            with patch(
                "rag.retrieval.index.load_json_to_sqlite", side_effect=fake_loader
            ) as loader, patch(
                "rag.retrieval.index.build_chunks", return_value=chunks
            ) as build_chunks, patch(
                "rag.retrieval.index.insert_embeddings"
            ) as insert_embeddings:
                first_path = ensure_index(
                    source_path,
                    artifact_dir=artifact_dir,
                    embed_texts_callable=fake_embed,
                    embedding_model_identity="model-a",
                )
                second_path = ensure_index(
                    source_path,
                    artifact_dir=artifact_dir,
                    embed_texts_callable=fake_embed,
                    embedding_model_identity="model-a",
                )

            self.assertEqual(first_path, second_path)
            self.assertEqual(first_path.parent, artifact_dir.resolve())
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(build_chunks.call_count, 1)
            self.assertEqual(insert_embeddings.call_count, 1)
            self.assertEqual(embed_calls, [["C100 first"]])

            with closing(sqlite3.connect(first_path)) as connection:
                metadata = connection.execute(
                    """
                    SELECT source_json_fingerprint, embedding_model_identity,
                           vector_dimension, chunk_count
                    FROM semantic_index_metadata
                    """
                ).fetchone()
                stored_chunk = connection.execute(
                    "SELECT chunk_id, text FROM semantic_chunks"
                ).fetchone()

            self.assertEqual(metadata[1:], ("model-a", 384, 1))
            self.assertEqual(stored_chunk, ("chunk-1", "C100 first"))

    def test_query_reuses_index_and_embeds_only_the_question(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "curriculum.json"
            artifact_dir = directory_path / "rag_artifacts"
            self._write_source(source_path)
            chunks = [
                {
                    "chunk_id": "chunk-1",
                    "text": "C100 first",
                    "provenance": [{"source_page": 7}],
                }
            ]
            embed_calls = []

            def fake_embed(texts):
                values = list(texts)
                embed_calls.append(values)
                return [[0.0] * 384 for _ in values]

            def fake_loader(_source, output):
                Path(output).touch()

            with patch(
                "rag.retrieval.index.load_json_to_sqlite", side_effect=fake_loader
            ) as loader, patch(
                "rag.retrieval.index.build_chunks", return_value=chunks
            ), patch("rag.retrieval.index.insert_embeddings") as insert_embeddings, patch(
                "rag.retrieval.index.nearest_neighbor_search",
                return_value=[{"chunk_id": "chunk-1", "distance": 0.25}],
            ):
                first = query_index(
                    source_path,
                    "วิชาอะไร",
                    artifact_dir=artifact_dir,
                    embed_texts_callable=fake_embed,
                    embedding_model_identity="model-a",
                )
                second = query_index(
                    source_path,
                    "วิชาอะไร",
                    artifact_dir=artifact_dir,
                    embed_texts_callable=fake_embed,
                    embedding_model_identity="model-a",
                )

            self.assertEqual(first, second)
            self.assertEqual(embed_calls, [["C100 first"], ["วิชาอะไร"], ["วิชาอะไร"]])
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(insert_embeddings.call_count, 1)
            self.assertEqual(second[0]["source_page"], [7])

    def test_rebuilds_when_source_model_or_dimension_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "curriculum.json"
            artifact_dir = directory_path / "rag_artifacts"
            self._write_source(source_path, "first")

            def fake_embed(texts):
                return [[0.0] * 384 for _ in texts]

            def fake_loader(_source, output):
                Path(output).touch()

            with patch(
                "rag.retrieval.index.load_json_to_sqlite", side_effect=fake_loader
            ) as loader, patch(
                "rag.retrieval.index.build_chunks",
                return_value=[{"chunk_id": "chunk-1", "text": "C100"}],
            ), patch("rag.retrieval.index.insert_embeddings") as insert_embeddings:
                ensure_index(
                    source_path,
                    artifact_dir=artifact_dir,
                    embed_texts_callable=fake_embed,
                    embedding_model_identity="model-a",
                )
                self._write_source(source_path, "changed")
                ensure_index(
                    source_path,
                    artifact_dir=artifact_dir,
                    embed_texts_callable=fake_embed,
                    embedding_model_identity="model-a",
                )
                ensure_index(
                    source_path,
                    artifact_dir=artifact_dir,
                    embed_texts_callable=fake_embed,
                    embedding_model_identity="model-b",
                )
                ensure_index(
                    source_path,
                    artifact_dir=artifact_dir,
                    embed_texts_callable=fake_embed,
                    embedding_model_identity="model-b",
                    vector_dimension=128,
                )

            self.assertEqual(loader.call_count, 4)
            self.assertEqual(insert_embeddings.call_count, 4)


if __name__ == "__main__":
    unittest.main()
