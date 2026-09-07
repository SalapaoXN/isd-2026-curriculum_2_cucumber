import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rag.build_index import main as build_index_main
from rag.retrieval.index import ensure_index, query_index


class RagIndexTest(unittest.TestCase):
    def _write_source(self, path: Path, marker: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "program": "PROGRAM",
                    "plan": "regular",
                    "courses": [{"code": "C100", "name_th": marker}],
                    "source_provenance": [
                        {
                            "source_filename": f"{path.stem}.pdf",
                            "source_page": 7,
                            "document_category": "plan",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _fake_loader(_source: Path, output: Path) -> None:
        output.touch()

    @staticmethod
    def _fake_embeddings(texts):
        return [[0.0] * 384 for _ in texts]

    def test_combines_sources_and_persists_chunk_metadata_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_a = directory_path / "a.json"
            source_b = directory_path / "b.json"
            artifact_dir = directory_path / "rag_artifacts"
            self._write_source(source_a, "course A")
            self._write_source(source_b, "course B")
            build_calls = []

            def fake_build(_database):
                source_name = "a" if len(build_calls) == 0 else "b"
                build_calls.append(source_name)
                return [
                    {
                        "chunk_id": f"chunk-{source_name}",
                        "text": f"text {source_name}",
                        "course_code": "C100",
                        "provenance": [
                            {
                                "source_document_key": f"{source_name}.pdf",
                                "source_page": 7,
                            }
                        ],
                    }
                ]

            with patch(
                "rag.retrieval.index.load_json_to_sqlite",
                side_effect=self._fake_loader,
            ) as loader, patch(
                "rag.retrieval.index.build_chunks", side_effect=fake_build
            ), patch("rag.retrieval.index.insert_embeddings") as insert_embeddings:
                index_path = ensure_index(
                    [source_b, source_a],
                    artifact_dir=artifact_dir,
                    embed_texts_callable=self._fake_embeddings,
                    embedding_model_identity="model-a",
                )

            self.assertEqual(index_path, artifact_dir.resolve() / "semantic.db")
            self.assertTrue(index_path.is_file())
            self.assertEqual(loader.call_count, 2)
            self.assertEqual(build_calls, ["a", "b"])
            insert_embeddings.assert_called_once()

            with closing(sqlite3.connect(index_path)) as connection:
                source_rows = connection.execute(
                    """
                    SELECT source_file_identity, source_filename,
                           source_json_fingerprint
                    FROM semantic_index_sources
                    ORDER BY source_file_identity
                    """
                ).fetchall()
                metadata = connection.execute(
                    """
                    SELECT source_json_fingerprint, embedding_model_identity,
                           vector_dimension, chunk_count
                    FROM semantic_index_metadata
                    """
                ).fetchone()
                chunks = connection.execute(
                    "SELECT chunk_id, chunk_json, source_file_identity FROM semantic_chunks"
                ).fetchall()

            self.assertEqual(len(source_rows), 2)
            self.assertEqual(
                [row[2] for row in source_rows],
                [
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (source_a, source_b)
                ],
            )
            self.assertEqual(metadata[1:], ("model-a", 384, 2))
            self.assertEqual(len(chunks), 2)
            stored = {row[0]: json.loads(row[1]) for row in chunks}
            self.assertEqual(stored["chunk-a"]["program"], "PROGRAM")
            self.assertEqual(stored["chunk-a"]["plan"], "regular")
            self.assertEqual(stored["chunk-a"]["course_code"], "C100")
            self.assertEqual(stored["chunk-a"]["source_page"], [7])
            self.assertEqual(stored["chunk-a"]["source_document_key"], "a.pdf")
            self.assertEqual(
                stored["chunk-a"]["source_file_identity"], str(source_a.resolve())
            )

    def test_valid_index_is_reused_and_query_embeds_only_question(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "curriculum.json"
            artifact_dir = directory_path / "rag_artifacts"
            self._write_source(source_path, "course")
            embed_calls = []
            chunks = [
                {
                    "chunk_id": "chunk-1",
                    "text": "indexed curriculum content",
                    "provenance": [{"source_page": 12}],
                }
            ]

            def fake_embed(texts):
                values = list(texts)
                embed_calls.append(values)
                return [[0.0] * 384 for _ in values]

            with patch(
                "rag.retrieval.index.load_json_to_sqlite",
                side_effect=self._fake_loader,
            ) as loader, patch(
                "rag.retrieval.index.build_chunks", return_value=chunks
            ) as build_chunks, patch(
                "rag.retrieval.index.insert_embeddings"
            ) as insert_embeddings, patch(
                "rag.retrieval.index.nearest_neighbor_search",
                return_value=[{"chunk_id": "chunk-1", "distance": 0.1}],
            ):
                first = query_index(
                    source_path,
                    "คำถามแรก",
                    artifact_dir=artifact_dir,
                    embed_texts_callable=fake_embed,
                    embedding_model_identity="model-a",
                )
                second = query_index(
                    source_path,
                    "คำถามที่สอง",
                    artifact_dir=artifact_dir,
                    embed_texts_callable=fake_embed,
                    embedding_model_identity="model-a",
                )

            self.assertEqual(first[0]["source_page"], [12])
            self.assertEqual(second[0]["source_page"], [12])
            self.assertEqual(
                embed_calls,
                [["indexed curriculum content"], ["คำถามแรก"], ["คำถามที่สอง"]],
            )
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(build_chunks.call_count, 1)
            self.assertEqual(insert_embeddings.call_count, 1)

    def test_rebuilds_when_any_source_model_or_dimension_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_a = directory_path / "a.json"
            source_b = directory_path / "b.json"
            artifact_dir = directory_path / "rag_artifacts"
            self._write_source(source_a, "a")
            self._write_source(source_b, "b")

            with patch(
                "rag.retrieval.index.load_json_to_sqlite",
                side_effect=self._fake_loader,
            ) as loader, patch(
                "rag.retrieval.index.build_chunks",
                return_value=[{"chunk_id": "chunk", "text": "text"}],
            ) as build_chunks, patch(
                "rag.retrieval.index.insert_embeddings"
            ) as insert_embeddings:
                ensure_index(
                    [source_a, source_b],
                    artifact_dir=artifact_dir,
                    embed_texts_callable=self._fake_embeddings,
                    embedding_model_identity="model-a",
                )
                ensure_index(
                    [source_a, source_b],
                    artifact_dir=artifact_dir,
                    embed_texts_callable=self._fake_embeddings,
                    embedding_model_identity="model-a",
                )
                self._write_source(source_b, "changed")
                ensure_index(
                    [source_a, source_b],
                    artifact_dir=artifact_dir,
                    embed_texts_callable=self._fake_embeddings,
                    embedding_model_identity="model-a",
                )
                ensure_index(
                    [source_a, source_b],
                    artifact_dir=artifact_dir,
                    embed_texts_callable=self._fake_embeddings,
                    embedding_model_identity="model-b",
                )
                ensure_index(
                    [source_a, source_b],
                    artifact_dir=artifact_dir,
                    embed_texts_callable=self._fake_embeddings,
                    embedding_model_identity="model-b",
                    vector_dimension=128,
                )

            self.assertEqual(loader.call_count, 8)
            self.assertEqual(build_chunks.call_count, 8)
            self.assertEqual(insert_embeddings.call_count, 4)

    def test_build_index_cli_accepts_multiple_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_a = directory_path / "a.json"
            source_b = directory_path / "b.json"
            self._write_source(source_a, "a")
            self._write_source(source_b, "b")
            output = io.StringIO()

            with patch(
                "rag.build_index.ensure_index",
                return_value=Path("rag_artifacts/semantic.db"),
            ) as ensure:
                with redirect_stdout(output):
                    build_index_main([str(source_a), str(source_b)])

            ensure.assert_called_once_with(
                [source_a, source_b],
                index_path=Path("rag_artifacts") / "semantic.db",
            )


if __name__ == "__main__":
    unittest.main()
