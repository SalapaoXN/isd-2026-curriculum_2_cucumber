import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rag.retrieval import vector_store


class _Cursor:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, search_rows=()):
        self.execute_calls = []
        self.executemany_calls = []
        self.search_rows = search_rows

    def enable_load_extension(self, enabled):
        return None

    def execute(self, sql, parameters=()):
        self.execute_calls.append((sql, parameters))
        if "SELECT chunk_id, distance" in sql:
            return _Cursor(self.search_rows)
        return _Cursor()

    def executemany(self, sql, rows):
        self.executemany_calls.append((sql, list(rows)))

    def commit(self):
        return None

    def close(self):
        return None


class RagVectorStoreTest(unittest.TestCase):
    def test_creates_inserts_by_chunk_id_and_searches_distance(self):
        vector = np.arange(vector_store.EMBEDDING_DIMENSION, dtype=np.float32)
        connection = _Connection(search_rows=[("course-1-metadata", 0.125)])

        with patch.object(vector_store, "_load_sqlite_vec") as load_extension:
            with patch.object(vector_store.sqlite3, "connect", return_value=connection):
                vector_store.create_vector_table("rag.db")
                vector_store.insert_embeddings(
                    "rag.db", [{"chunk_id": "course-1-metadata"}], [vector]
                )
                results = vector_store.nearest_neighbor_search("rag.db", vector, limit=1)

        load_extension.assert_called()
        create_sql = connection.execute_calls[0][0]
        self.assertIn("float[384]", create_sql)
        self.assertIn("+chunk_id TEXT", create_sql)
        insert_sql, rows = connection.executemany_calls[0]
        self.assertIn("chunk_id", insert_sql)
        self.assertEqual(rows[0][1], "course-1-metadata")
        self.assertEqual(rows[0][0], vector.tobytes())
        self.assertEqual(results, [{"chunk_id": "course-1-metadata", "distance": 0.125}])

    def test_direct_vector_reads_load_real_vec0_connections(self):
        directory = tempfile.TemporaryDirectory()
        database_path = f"{directory.name}/curriculum.db"
        left = np.zeros(vector_store.EMBEDDING_DIMENSION, dtype=np.float32)
        left[0] = 1.0
        right = np.zeros(vector_store.EMBEDDING_DIMENSION, dtype=np.float32)
        right[1] = 1.0
        try:
            vector_store.create_vector_table(database_path)
            vector_store.insert_embeddings(
                database_path,
                [{"chunk_id": "left"}, {"chunk_id": "right"}],
                [left, right],
            )
            with patch.object(
                vector_store,
                "_load_sqlite_vec",
                wraps=vector_store._load_sqlite_vec,
            ) as load_extension:
                scores = vector_store.score_candidate_vectors(
                    database_path, left, ["left", "right"]
                )
                comparison = vector_store.compare_stored_vectors(
                    database_path, "left", "right"
                )

            self.assertEqual([item["chunk_id"] for item in scores["scores"]], [
                "left",
                "right",
            ])
            self.assertEqual(comparison["status"], "complete")
            self.assertEqual(comparison["cosine_distance"], 1.0)
            self.assertEqual(comparison["cosine_similarity"], 0.0)
            self.assertEqual(load_extension.call_count, 2)
        finally:
            directory.cleanup()


if __name__ == "__main__":
    unittest.main()
