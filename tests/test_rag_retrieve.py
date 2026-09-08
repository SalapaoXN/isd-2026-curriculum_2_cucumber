import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

import numpy as np

from rag.retrieval.retrieve import retrieve


class RagRetrieveTest(unittest.TestCase):
    def _semantic_chunk_database(self, course_codes):
        directory = tempfile.TemporaryDirectory()
        database_path = f"{directory.name}/curriculum.db"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                "CREATE TABLE semantic_chunks (chunk_id TEXT, chunk_json TEXT)"
            )
            connection.executemany(
                "INSERT INTO semantic_chunks VALUES (?, ?)",
                [
                    (
                        f"chunk-{index}",
                        json.dumps({"course_code": code, "text": f"text {index}"}),
                    )
                    for index, code in enumerate(course_codes)
                ],
            )
            connection.commit()
        return directory, database_path

    def test_embeds_query_and_returns_ranked_cited_evidence(self):
        query_embedding = np.zeros(384, dtype=np.float32)
        provenance = {
            "provenance_id": 7,
            "source_filename": "page-42.png",
            "source_page": 42,
            "document_category": "plan",
        }
        search_results = [
            {
                "chunk_id": "course-2-placement-2-metadata",
                "distance": 0.4,
                "text": "second evidence",
                "provenance": [provenance],
            },
            {
                "chunk_id": "course-1-placement-1-metadata",
                "distance": 0.2,
                "text": "first evidence",
                "provenance": [provenance],
            },
        ]

        with patch(
            "rag.retrieval.retrieve.embed_texts", return_value=np.array([query_embedding])
        ) as embed_mock, patch(
            "rag.retrieval.retrieve.search", return_value=search_results
        ) as search_mock:
            results = retrieve("curriculum.db", "วิชาเรียน", k=2)

        embed_mock.assert_called_once_with(["วิชาเรียน"])
        search_mock.assert_called_once()
        search_call = search_mock.call_args
        self.assertEqual(search_call.args[0], "curriculum.db")
        np.testing.assert_array_equal(search_call.args[1], query_embedding)
        self.assertEqual(search_call.kwargs, {"k": 2})
        self.assertEqual(
            results,
            [
                {
                    "chunk_id": "course-1-placement-1-metadata",
                    "text": "first evidence",
                    "distance": 0.2,
                    "source_page": [42],
                    "provenance": [provenance],
                },
                {
                    "chunk_id": "course-2-placement-2-metadata",
                    "text": "second evidence",
                    "distance": 0.4,
                    "source_page": [42],
                    "provenance": [provenance],
                },
            ],
        )

    def test_explicit_course_code_restricts_results_before_top_k(self):
        directory, database_path = self._semantic_chunk_database(
            ["99999999", "06016418", "06016418"]
        )
        query_embedding = np.zeros(384, dtype=np.float32)
        search_results = [
            {"chunk_id": "chunk-0", "distance": 0.1, "text": "unrelated", "provenance": []},
            {"chunk_id": "chunk-1", "distance": 0.2, "text": "target one", "provenance": []},
            {"chunk_id": "chunk-2", "distance": 0.3, "text": "target two", "provenance": []},
        ]
        try:
            with patch(
                "rag.retrieval.retrieve.embed_texts", return_value=np.array([query_embedding])
            ), patch(
                "rag.retrieval.retrieve.search", return_value=search_results
            ) as search_mock:
                results = retrieve(database_path, "ขอรายละเอียดวิชา 06016418", k=1)

            self.assertEqual([result["chunk_id"] for result in results], ["chunk-1"])
            self.assertEqual(search_mock.call_args.kwargs, {"k": 3})
        finally:
            directory.cleanup()

    def test_unknown_explicit_course_code_returns_no_unrelated_evidence(self):
        directory, database_path = self._semantic_chunk_database(["06016418"])
        try:
            with patch("rag.retrieval.retrieve.embed_texts") as embed_mock, patch(
                "rag.retrieval.retrieve.search"
            ) as search_mock:
                results = retrieve(database_path, "มีวิชา 12345678 ไหม", k=10)

            self.assertEqual(results, [])
            embed_mock.assert_not_called()
            search_mock.assert_not_called()
        finally:
            directory.cleanup()


if __name__ == "__main__":
    unittest.main()
