import hashlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rag.build_index import main as build_index_main
from rag.retrieval import index as index_module
from rag.retrieval.index import (
    ARTIFACTS_DIR,
    DEFAULT_INDEX_NAME,
    ensure_index,
    index_path_for_source,
    query_index,
)


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
    def _fake_embeddings(texts):
        return [[0.0] * 384 for _ in texts]

    def test_default_runtime_database_never_uses_legacy_artifacts(self):
        project_root = Path(__file__).resolve().parents[1]
        expected_directory = project_root / "cucumber_outputs" / "runtime"

        self.assertEqual(ARTIFACTS_DIR, expected_directory)
        self.assertEqual(
            index_path_for_source(Path("curriculum.json")),
            expected_directory / DEFAULT_INDEX_NAME,
        )
        self.assertNotIn("rag_artifacts", str(ARTIFACTS_DIR))

    def test_combines_sources_and_persists_chunk_metadata_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_a = directory_path / "a.json"
            source_b = directory_path / "b.json"
            artifact_dir = directory_path / "cucumber_outputs" / "runtime"
            self._write_source(source_a, "course A")
            self._write_source(source_b, "course B")
            with patch("rag.retrieval.index.insert_embeddings") as insert_embeddings:
                index_path = ensure_index(
                    [source_b, source_a],
                    artifact_dir=artifact_dir,
                    embed_texts_callable=self._fake_embeddings,
                    embedding_model_identity="model-a",
                )

            self.assertEqual(index_path, artifact_dir.resolve() / "curriculum.db")
            self.assertTrue(index_path.is_file())
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
                relational_counts = connection.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM catalogs),
                        (SELECT COUNT(*) FROM courses),
                        (SELECT COUNT(*) FROM curriculum_plans),
                        (SELECT COUNT(*) FROM plan_placements)
                    """
                ).fetchone()

            self.assertEqual(len(source_rows), 2)
            expected_source_rows = []
            for path in (source_a, source_b):
                fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
                expected_source_rows.append(
                    (f"content:{fingerprint}", path.name, fingerprint)
                )
            self.assertEqual(
                source_rows,
                sorted(expected_source_rows),
            )
            self.assertEqual(metadata[1:], ("model-a", 384, 4))
            self.assertEqual(len(chunks), 4)
            self.assertEqual(relational_counts, (2, 2, 2, 2))
            stored = {row[0]: json.loads(row[1]) for row in chunks}
            chunk_a = next(
                chunk
                for chunk in stored.values()
                if chunk["source_filename"] == "a.json"
                and chunk["chunk_type"] == "metadata"
            )
            self.assertEqual(chunk_a["program"], "PROGRAM")
            self.assertEqual(chunk_a["plan"], "regular")
            self.assertEqual(chunk_a["course_code"], "C100")
            self.assertEqual(chunk_a["source_page"], [7])
            self.assertEqual(chunk_a["source_document_key"], "a.pdf")
            self.assertEqual(
                chunk_a["source_file_identity"],
                f"content:{hashlib.sha256(source_a.read_bytes()).hexdigest()}",
            )

    def test_source_identity_is_portable(self):
        project_root = Path(__file__).resolve().parents[1]
        canonical_path = (
            project_root
            / "outputs"
            / "consolidated"
            / "ait"
            / "full"
            / "merged_ait_no_plan_full.json"
        )

        with patch("rag.retrieval.index._PROJECT_ROOT", project_root):
            canonical_identity = index_module._source_identity(
                canonical_path, "fingerprint"
            )
            path_style_identity = index_module._source_identity(
                Path(str(canonical_path).replace("\\", "/")), "fingerprint"
            )

        self.assertEqual(
            canonical_identity,
            "repo:outputs/consolidated/ait/full/merged_ait_no_plan_full.json",
        )
        self.assertEqual(path_style_identity, canonical_identity)
        self.assertNotIn("\\", canonical_identity)

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first" / "source.json"
            second = Path(directory) / "second" / "source.json"
            fingerprint = "a" * 64
            self.assertEqual(
                index_module._source_identity(first, fingerprint),
                "content:" + fingerprint,
            )
            self.assertEqual(
                index_module._source_identity(second, fingerprint),
                "content:" + fingerprint,
            )

    def test_reuses_index_when_repository_root_moves(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            root_a = directory_path / "repo-a"
            root_b = directory_path / "repo-b"
            relative_source = Path(
                "outputs",
                "consolidated",
                "program",
                "full",
                "merged_program_full.json",
            )
            source_a = root_a / relative_source
            source_b = root_b / relative_source
            source_a.parent.mkdir(parents=True)
            source_b.parent.mkdir(parents=True)
            self._write_source(source_a, "course")
            self._write_source(source_b, "course")
            artifact_dir = root_a / "cucumber_outputs" / "runtime"

            with patch(
                "rag.retrieval.index._PROJECT_ROOT", root_a
            ), patch(
                "rag.retrieval.index.build_chunks",
                return_value=[{"chunk_id": "chunk", "text": "text", "course_id": 1}],
            ) as build_chunks, patch(
                "rag.retrieval.index.insert_embeddings"
            ) as insert_embeddings:
                ensure_index(
                    source_a,
                    artifact_dir=artifact_dir,
                    embed_texts_callable=self._fake_embeddings,
                    embedding_model_identity="model-a",
                )
                with patch("rag.retrieval.index._PROJECT_ROOT", root_b):
                    ensure_index(
                        source_b,
                        artifact_dir=artifact_dir,
                        embed_texts_callable=self._fake_embeddings,
                        embedding_model_identity="model-a",
                    )

            self.assertEqual(build_chunks.call_count, 1)
            self.assertEqual(insert_embeddings.call_count, 1)
            with closing(sqlite3.connect(artifact_dir / DEFAULT_INDEX_NAME)) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT source_file_identity FROM semantic_index_sources"
                    ).fetchone()[0],
                    "repo:outputs/consolidated/program/full/merged_program_full.json",
                )

    def test_valid_index_is_reused_and_query_embeds_only_question(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "curriculum.json"
            artifact_dir = directory_path / "cucumber_outputs" / "runtime"
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
            self.assertEqual(build_chunks.call_count, 1)
            self.assertEqual(insert_embeddings.call_count, 1)

    def test_rebuilds_when_any_source_model_or_dimension_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_a = directory_path / "a.json"
            source_b = directory_path / "b.json"
            artifact_dir = directory_path / "cucumber_outputs" / "runtime"
            self._write_source(source_a, "a")
            self._write_source(source_b, "b")

            with patch(
                "rag.retrieval.index.build_chunks",
                return_value=[{"chunk_id": "chunk", "text": "text", "course_id": 1}],
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

            self.assertEqual(build_chunks.call_count, 4)
            self.assertEqual(insert_embeddings.call_count, 4)

    def test_build_index_cli_accepts_multiple_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_a = directory_path / "a.json"
            source_b = directory_path / "b.json"
            self._write_source(source_a, "a")
            self._write_source(source_b, "b")
            output = io.StringIO()

            default_index_path = ARTIFACTS_DIR / DEFAULT_INDEX_NAME
            with patch(
                "rag.build_index.ensure_index",
                return_value=default_index_path,
            ) as ensure:
                with redirect_stdout(output):
                    build_index_main([str(source_a), str(source_b)])

            ensure.assert_called_once_with(
                [source_a, source_b],
                index_path=default_index_path,
            )

    def test_cli_loads_dotenv_before_embedding_index_starts(self):
        events = []

        def fake_load_dotenv():
            events.append("dotenv")
            os.environ["HF_TOKEN"] = "dotenv-token"

        def fake_ensure_index(_paths, index_path):
            events.append(os.environ.get("HF_TOKEN"))
            return Path(index_path)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            with patch(
                "rag.build_index.load_dotenv", side_effect=fake_load_dotenv
            ) as load_dotenv, patch(
                "rag.build_index.ensure_index", side_effect=fake_ensure_index
            ):
                with redirect_stdout(io.StringIO()):
                    build_index_main(["curriculum.json"])

        load_dotenv.assert_called_once_with()
        self.assertEqual(events, ["dotenv", "dotenv-token"])

    def test_cli_preserves_explicit_hf_token(self):
        def dotenv_without_override():
            os.environ.setdefault("HF_TOKEN", "dotenv-token")

        with patch.dict(os.environ, {"HF_TOKEN": "explicit-token"}):
            with patch(
                "rag.build_index.load_dotenv", side_effect=dotenv_without_override
            ) as load_dotenv, patch(
                "rag.build_index.ensure_index",
                return_value=ARTIFACTS_DIR / DEFAULT_INDEX_NAME,
            ) as ensure:
                with redirect_stdout(io.StringIO()):
                    build_index_main(["curriculum.json"])

            load_dotenv.assert_called_once_with()
            self.assertEqual(os.environ["HF_TOKEN"], "explicit-token")
            ensure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
