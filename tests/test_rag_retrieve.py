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

    def _named_semantic_chunk_database(self, courses):
        directory = tempfile.TemporaryDirectory()
        database_path = f"{directory.name}/curriculum.db"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                "CREATE TABLE courses "
                "(course_code_normalized TEXT, name_th TEXT, name_en TEXT)"
            )
            connection.executemany(
                "INSERT INTO courses VALUES (?, ?, ?)", courses
            )
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
                    for index, code in enumerate(
                        [course[0] for course in courses] + ["99999999"]
                    )
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

    def test_multiple_explicit_course_codes_use_their_union(self):
        directory, database_path = self._semantic_chunk_database(
            ["99999999", "06016402", "06016402", "06026207"]
        )
        query_embedding = np.zeros(384, dtype=np.float32)
        search_results = [
            {"chunk_id": "chunk-0", "distance": 0.1, "text": "unrelated", "provenance": []},
            {"chunk_id": "chunk-1", "distance": 0.2, "text": "first course", "provenance": []},
            {"chunk_id": "chunk-2", "distance": 0.3, "text": "first course detail", "provenance": []},
            {"chunk_id": "chunk-3", "distance": 0.4, "text": "second course", "provenance": []},
        ]
        try:
            with patch(
                "rag.retrieval.retrieve.embed_texts", return_value=np.array([query_embedding])
            ), patch(
                "rag.retrieval.retrieve.search", return_value=search_results
            ) as search_mock:
                results = retrieve(
                    database_path,
                    "เปรียบเทียบวิชา 06016402 กับ 06026207 และ 06016402",
                    k=3,
                )

            self.assertEqual(
                [result["chunk_id"] for result in results],
                ["chunk-1", "chunk-2", "chunk-3"],
            )
            self.assertEqual(search_mock.call_args.kwargs, {"k": 4})
        finally:
            directory.cleanup()

    def test_exact_course_name_restricts_results_to_its_code(self):
        directory, database_path = self._named_semantic_chunk_database(
            [("06016402", "พื้นฐาน", "INFORMATION TECHNOLOGY FUNDAMENTALS")]
        )
        query_embedding = np.zeros(384, dtype=np.float32)
        search_results = [
            {"chunk_id": "chunk-1", "distance": 0.1, "text": "unrelated", "provenance": []},
            {
                "chunk_id": "chunk-0",
                "distance": 0.2,
                "text": "target description",
                "provenance": [{"source_page": 328}],
            },
            {"chunk_id": "chunk-2", "distance": 0.3, "text": "other", "provenance": []},
        ]
        try:
            with patch(
                "rag.retrieval.retrieve.embed_texts", return_value=np.array([query_embedding])
            ), patch(
                "rag.retrieval.retrieve.search", return_value=search_results
            ) as search_mock:
                results = retrieve(
                    database_path,
                    "INFORMATION TECHNOLOGY FUNDAMENTALS เรียนเกี่ยวกับอะไรบ้าง?",
                    k=1,
                )

            self.assertEqual([result["chunk_id"] for result in results], ["chunk-0"])
            self.assertEqual(results[0]["provenance"], [{"source_page": 328}])
            self.assertEqual(search_mock.call_args.kwargs, {"k": 2})
        finally:
            directory.cleanup()

    def test_ambiguous_exact_course_name_keeps_global_search(self):
        directory, database_path = self._named_semantic_chunk_database(
            [
                ("06016402", "", "SHARED COURSE"),
                ("06026207", "", "SHARED COURSE"),
            ]
        )
        query_embedding = np.zeros(384, dtype=np.float32)
        search_results = [
            {"chunk_id": "chunk-2", "distance": 0.1, "text": "unrelated", "provenance": []},
            {"chunk_id": "chunk-0", "distance": 0.2, "text": "first", "provenance": []},
        ]
        try:
            with patch(
                "rag.retrieval.retrieve.embed_texts", return_value=np.array([query_embedding])
            ), patch(
                "rag.retrieval.retrieve.search", return_value=search_results
            ) as search_mock:
                results = retrieve(database_path, "SHARED COURSE topics", k=1)

            self.assertEqual([result["chunk_id"] for result in results], ["chunk-2"])
            self.assertEqual(search_mock.call_args.kwargs, {"k": 1})
        finally:
            directory.cleanup()


if __name__ == "__main__":
    unittest.main()
