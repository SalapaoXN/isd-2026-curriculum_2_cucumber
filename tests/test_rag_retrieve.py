import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

import numpy as np

from rag.retrieval.retrieve import (
    CONSTRAINED_RETRIEVAL_STATES,
    CONSTRAINED_TOPIC_DISTANCE_THRESHOLD,
    ConstrainedTopicRetrievalResult,
    enrich_candidate_description_scores,
    fetch_course_description_evidence,
    make_constrained_topic_retrieval_result,
    map_course_candidates_to_description_evidence,
    lexical_topic_match,
    retrieve,
    retrieve_constrained_topic_evidence,
)
from rag.retrieval.vector_store import score_candidate_vectors


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

    def _course_chunk_database(self, chunks):
        directory = tempfile.TemporaryDirectory()
        database_path = f"{directory.name}/curriculum.db"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                "CREATE TABLE semantic_chunks (chunk_id TEXT, chunk_json TEXT)"
            )
            connection.executemany(
                "INSERT INTO semantic_chunks VALUES (?, ?)",
                [(chunk_id, json.dumps(chunk)) for chunk_id, chunk in chunks],
            )
            connection.commit()
        return directory, database_path

    def _vector_database(self, vectors):
        directory = tempfile.TemporaryDirectory()
        database_path = f"{directory.name}/curriculum.db"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                "CREATE TABLE vector (chunk_id TEXT, embed_byte BLOB)"
            )
            connection.executemany(
                "INSERT INTO vector VALUES (?, ?)",
                [(chunk_id, np.asarray(vector, dtype=np.float32).tobytes())
                 for chunk_id, vector in vectors],
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

    def test_exact_course_description_fetch_preserves_identity_text_and_provenance(self):
        provenance = {"source_page": 328, "source_filename": "page-328.png"}
        directory, database_path = self._course_chunk_database(
            [
                (
                    "description-target",
                    {
                        "chunk_type": "description",
                        "course_id": 42,
                        "course_code": "06016402",
                        "program": "IT",
                        "text": "target description",
                        "provenance": [provenance],
                    },
                ),
                (
                    "description-other-course",
                    {
                        "chunk_type": "description",
                        "course_id": 99,
                        "course_code": "06016499",
                        "text": "other description",
                        "provenance": [],
                    },
                ),
            ]
        )
        try:
            with patch("rag.retrieval.retrieve.embed_texts") as embed_mock, patch(
                "rag.retrieval.retrieve.search"
            ) as search_mock:
                results = fetch_course_description_evidence(database_path, 42)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["chunk_id"], "description-target")
            self.assertEqual(results[0]["course_id"], 42)
            self.assertEqual(results[0]["course_code"], "06016402")
            self.assertEqual(results[0]["text"], "target description")
            self.assertIsNone(results[0]["distance"])
            self.assertEqual(results[0]["provenance"], [provenance])
            self.assertEqual(results[0]["source_page"], [328])
            embed_mock.assert_not_called()
            search_mock.assert_not_called()
        finally:
            directory.cleanup()

    def test_exact_course_description_fetch_excludes_metadata_and_wrong_course(self):
        directory, database_path = self._course_chunk_database(
            [
                (
                    "target-metadata",
                    {"chunk_type": "metadata", "course_id": 42, "text": "metadata"},
                ),
                (
                    "other-description",
                    {
                        "chunk_type": "description",
                        "course_id": 43,
                        "text": "wrong course",
                    },
                ),
            ]
        )
        try:
            self.assertEqual(fetch_course_description_evidence(database_path, 42), [])
        finally:
            directory.cleanup()

    def test_exact_course_description_fetch_returns_empty_when_description_missing(self):
        directory, database_path = self._course_chunk_database(
            [("target-metadata", {"chunk_type": "metadata", "course_id": 42})]
        )
        try:
            self.assertEqual(fetch_course_description_evidence(database_path, 42), [])
            self.assertEqual(fetch_course_description_evidence(database_path, 404), [])
        finally:
            directory.cleanup()

    def test_candidate_mapping_preserves_candidates_and_excludes_non_candidates(self):
        directory, database_path = self._course_chunk_database(
            [
                (
                    "description-one",
                    {
                        "chunk_type": "description",
                        "course_id": 1,
                        "course_code": "06016401",
                        "text": "one",
                        "provenance": [],
                    },
                ),
                (
                    "description-two",
                    {
                        "chunk_type": "description",
                        "course_id": 2,
                        "course_code": "06016402",
                        "text": "two",
                        "provenance": [],
                    },
                ),
                (
                    "description-not-candidate",
                    {
                        "chunk_type": "description",
                        "course_id": 99,
                        "course_code": "06016499",
                        "text": "outside",
                        "provenance": [],
                    },
                ),
            ]
        )
        try:
            candidates = [
                {"course_id": 1, "program": "IT", "partition": {"year": 1}},
                {"course_id": 2, "program": "IT", "partition": {"year": 2}},
            ]
            mapped = map_course_candidates_to_description_evidence(
                database_path, candidates
            )

            self.assertEqual([item["course_id"] for item in mapped], [1, 2])
            self.assertEqual(
                [item["description_evidence"][0]["text"] for item in mapped],
                ["one", "two"],
            )
            self.assertNotIn(99, [item["course_id"] for item in mapped])
            self.assertEqual(
                [item["partition"] for item in mapped],
                [{"year": 1}, {"year": 2}],
            )
        finally:
            directory.cleanup()

    def test_candidate_mapping_keeps_same_course_in_separate_partitions(self):
        directory, database_path = self._course_chunk_database(
            [
                (
                    "description-one",
                    {
                        "chunk_type": "description",
                        "course_id": 1,
                        "text": "one",
                        "provenance": [],
                    },
                )
            ]
        )
        try:
            mapped = map_course_candidates_to_description_evidence(
                database_path,
                (
                    {"course_id": 1, "partition": {"plan": "coop"}},
                    {"course_id": 1, "partition": {"plan": "no_coop"}},
                ),
            )

            self.assertEqual(len(mapped), 2)
            self.assertEqual(
                [item["partition"]["plan"] for item in mapped],
                ["coop", "no_coop"],
            )
            self.assertEqual(
                [item["description_evidence"][0]["text"] for item in mapped],
                ["one", "one"],
            )
        finally:
            directory.cleanup()

    def test_candidate_mapping_distinguishes_empty_candidates_from_missing_description(self):
        directory, database_path = self._course_chunk_database(
            [
                (
                    "metadata-only",
                    {"chunk_type": "metadata", "course_id": 3, "text": "metadata"},
                )
            ]
        )
        try:
            self.assertEqual(
                map_course_candidates_to_description_evidence(database_path, []), ()
            )
            mapped = map_course_candidates_to_description_evidence(
                database_path, [{"course_id": 3, "partition": {"year": 1}}]
            )
            self.assertEqual(len(mapped), 1)
            self.assertEqual(mapped[0]["description_evidence"], ())
        finally:
            directory.cleanup()

    def test_candidate_mapping_preserves_description_provenance(self):
        provenance = {"source_page": 12, "source_filename": "page-12.png"}
        directory, database_path = self._course_chunk_database(
            [
                (
                    "description-one",
                    {
                        "chunk_type": "description",
                        "course_id": 1,
                        "text": "one",
                        "provenance": [provenance],
                    },
                )
            ]
        )
        try:
            mapped = map_course_candidates_to_description_evidence(
                database_path, [{"course_id": 1}]
            )
            self.assertEqual(
                mapped[0]["description_evidence"][0]["provenance"],
                [provenance],
            )
        finally:
            directory.cleanup()

    def test_candidate_scoring_excludes_better_unrelated_global_vector(self):
        query = np.zeros(384, dtype=np.float32)
        query[0] = 1.0
        directory, database_path = self._vector_database(
            [
                ("candidate", query),
                ("unrelated", query),
            ]
        )
        try:
            with patch("rag.retrieval.vector_store.nearest_neighbor_search") as global_mock:
                result = score_candidate_vectors(
                    database_path, query, ["candidate"]
                )

            self.assertEqual([item["chunk_id"] for item in result["scores"]], ["candidate"])
            self.assertEqual(result["missing_chunk_ids"], [])
            self.assertEqual(result["invalid_chunk_ids"], [])
            global_mock.assert_not_called()
        finally:
            directory.cleanup()

    def test_candidate_scoring_scores_all_supplied_vectors_and_sorts_by_distance(self):
        query = np.zeros(384, dtype=np.float32)
        query[0] = 1.0
        orthogonal = np.zeros(384, dtype=np.float32)
        orthogonal[1] = 1.0
        opposite = -query
        directory, database_path = self._vector_database(
            [("opposite", opposite), ("orthogonal", orthogonal), ("same", query)]
        )
        try:
            result = score_candidate_vectors(
                database_path, query, ["opposite", "orthogonal", "same"]
            )

            self.assertEqual(
                [item["chunk_id"] for item in result["scores"]],
                ["same", "orthogonal", "opposite"],
            )
            self.assertEqual(
                [item["distance"] for item in result["scores"]],
                [0.0, 1.0, 2.0],
            )
            self.assertEqual(result["missing_chunk_ids"], [])
            self.assertEqual(result["invalid_chunk_ids"], [])
        finally:
            directory.cleanup()

    def test_candidate_scoring_breaks_equal_distance_ties_by_chunk_id(self):
        query = np.zeros(384, dtype=np.float32)
        query[0] = 1.0
        tie = np.zeros(384, dtype=np.float32)
        tie[1] = 1.0
        directory, database_path = self._vector_database(
            [("z-tie", tie), ("a-tie", tie)]
        )
        try:
            result = score_candidate_vectors(database_path, query, ["z-tie", "a-tie"])
            self.assertEqual(
                [item["chunk_id"] for item in result["scores"]],
                ["a-tie", "z-tie"],
            )
        finally:
            directory.cleanup()

    def test_candidate_scoring_reports_missing_ids(self):
        query = np.zeros(384, dtype=np.float32)
        query[0] = 1.0
        directory, database_path = self._vector_database([("present", query)])
        try:
            result = score_candidate_vectors(
                database_path, query, ["missing", "present"]
            )
            self.assertEqual(result["missing_chunk_ids"], ["missing"])
            self.assertEqual(
                [item["chunk_id"] for item in result["scores"]], ["present"]
            )
        finally:
            directory.cleanup()

    def test_candidate_scoring_empty_input_does_not_require_database(self):
        result = score_candidate_vectors(
            "missing.db", np.zeros(384, dtype=np.float32), []
        )
        self.assertEqual(
            result,
            {"scores": [], "missing_chunk_ids": [], "invalid_chunk_ids": []},
        )

    def test_candidate_scoring_reports_invalid_and_zero_norm_vectors(self):
        query = np.zeros(384, dtype=np.float32)
        query[0] = 1.0
        zero = np.zeros(384, dtype=np.float32)
        invalid = np.zeros(383, dtype=np.float32)
        directory = tempfile.TemporaryDirectory()
        database_path = f"{directory.name}/curriculum.db"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                "CREATE TABLE vector (chunk_id TEXT, embed_byte BLOB)"
            )
            connection.executemany(
                "INSERT INTO vector VALUES (?, ?)",
                [("zero", zero.tobytes()), ("invalid", invalid.tobytes())],
            )
            connection.commit()
        try:
            result = score_candidate_vectors(
                database_path, query, ["zero", "invalid"]
            )
            self.assertEqual(result["scores"], [])
            self.assertEqual(result["missing_chunk_ids"], [])
            self.assertEqual(result["invalid_chunk_ids"], ["zero", "invalid"])
            with self.assertRaises(ValueError):
                score_candidate_vectors(database_path, zero, ["zero"])
        finally:
            directory.cleanup()

    def test_description_score_enrichment_joins_scores_by_chunk_id(self):
        mapped = (
            {
                "course_id": 1,
                "program": "IT",
                "partition": {"year": 2},
                "description_evidence": (
                    {
                        "chunk_id": "description-one",
                        "text": "one",
                        "distance": None,
                        "provenance": [{"source_page": 10}],
                    },
                ),
            },
        )
        result = enrich_candidate_description_scores(
            mapped,
            {
                "scores": [{"chunk_id": "description-one", "distance": 0.25}],
                "missing_chunk_ids": [],
                "invalid_chunk_ids": [],
            },
        )

        candidate = result["candidates"][0]
        self.assertEqual(candidate["course_id"], 1)
        self.assertEqual(candidate["partition"], {"year": 2})
        self.assertEqual(
            candidate["description_evidence"][0]["chunk_id"],
            "description-one",
        )
        self.assertEqual(candidate["description_evidence"][0]["distance"], 0.25)
        self.assertEqual(
            candidate["description_evidence"][0]["provenance"],
            [{"source_page": 10}],
        )

    def test_description_score_enrichment_keeps_same_course_partitions_separate(self):
        mapped = (
            {
                "course_id": 1,
                "partition": {"plan": "coop"},
                "description_evidence": (
                    {"chunk_id": "description-one", "text": "one", "provenance": []},
                ),
            },
            {
                "course_id": 1,
                "partition": {"plan": "no_coop"},
                "description_evidence": (
                    {"chunk_id": "description-one", "text": "one", "provenance": []},
                ),
            },
        )
        result = enrich_candidate_description_scores(
            mapped,
            {
                "scores": [{"chunk_id": "description-one", "distance": 0.5}],
                "missing_chunk_ids": [],
                "invalid_chunk_ids": [],
            },
        )

        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(
            [candidate["partition"]["plan"] for candidate in result["candidates"]],
            ["coop", "no_coop"],
        )
        self.assertEqual(
            [
                candidate["description_evidence"][0]["distance"]
                for candidate in result["candidates"]
            ],
            [0.5, 0.5],
        )

    def test_description_score_enrichment_ignores_unrelated_scores_and_is_deterministic(self):
        mapped = (
            {
                "course_id": 2,
                "partition": {"year": 2},
                "description_evidence": (
                    {"chunk_id": "b", "text": "b", "provenance": []},
                    {"chunk_id": "a", "text": "a", "provenance": []},
                ),
            },
        )
        result = enrich_candidate_description_scores(
            mapped,
            {
                "scores": [
                    {"chunk_id": "unrelated", "distance": 0.01},
                    {"chunk_id": "a", "distance": 0.2},
                    {"chunk_id": "b", "distance": 0.2},
                ],
                "missing_chunk_ids": [],
                "invalid_chunk_ids": [],
            },
        )

        self.assertEqual(
            [item["chunk_id"] for item in result["candidates"][0]["description_evidence"]],
            ["b", "a"],
        )
        self.assertEqual(
            [item["distance"] for item in result["candidates"][0]["description_evidence"]],
            [0.2, 0.2],
        )
        self.assertNotIn(
            "unrelated",
            [item["chunk_id"] for item in result["candidates"][0]["description_evidence"]],
        )

    def test_description_score_enrichment_retains_missing_and_invalid_vector_state(self):
        mapped = (
            {
                "course_id": 3,
                "description_evidence": (
                    {"chunk_id": "missing", "text": "missing", "provenance": []},
                    {"chunk_id": "invalid", "text": "invalid", "provenance": []},
                ),
            },
        )
        result = enrich_candidate_description_scores(
            mapped,
            {
                "scores": [],
                "missing_chunk_ids": ["missing"],
                "invalid_chunk_ids": ["invalid"],
            },
        )

        self.assertEqual(result["missing_chunk_ids"], ["missing"])
        self.assertEqual(result["invalid_chunk_ids"], ["invalid"])
        self.assertEqual(
            [item["distance"] for item in result["candidates"][0]["description_evidence"]],
            [None, None],
        )

    def test_constrained_retrieval_result_distinguishes_empty_candidates(self):
        result = make_constrained_topic_retrieval_result([], {})
        self.assertEqual(result.status, "empty_structural_candidates")
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.scored_candidates, ())

    def test_constrained_retrieval_result_distinguishes_missing_descriptions(self):
        result = make_constrained_topic_retrieval_result(
            [{"course_id": 7, "partition": {"year": 2}, "description_evidence": ()}],
            {"scores": [], "missing_chunk_ids": [], "invalid_chunk_ids": []},
        )
        self.assertEqual(result.status, "description_missing")
        self.assertEqual(result.missing_description_course_ids, (7,))
        self.assertEqual(result.scored_candidates, ())

    def test_constrained_retrieval_result_distinguishes_missing_or_invalid_vectors(self):
        mapped = [
            {
                "course_id": 8,
                "partition": {"plan": "coop"},
                "description_evidence": (
                    {
                        "chunk_id": "description-eight",
                        "text": "eight",
                        "distance": None,
                        "provenance": [{"source_page": 8}],
                    },
                ),
            }
        ]
        result = make_constrained_topic_retrieval_result(
            mapped,
            {
                "scores": [],
                "missing_chunk_ids": ["description-eight"],
                "invalid_chunk_ids": [],
            },
        )
        self.assertEqual(result.status, "vector_missing_or_invalid")
        self.assertEqual(result.missing_vector_chunk_ids, ("description-eight",))
        self.assertEqual(result.candidates[0]["description_evidence"][0]["distance"], None)

    def test_constrained_retrieval_result_preserves_scored_evidence_and_provenance(self):
        mapped = [
            {
                "course_id": 9,
                "program": "IT",
                "partition": {"plan": "no_coop", "year": 3},
                "description_evidence": (
                    {
                        "chunk_id": "description-nine",
                        "text": "nine",
                        "distance": 0.3,
                        "provenance": [{"source_page": 9}],
                    },
                ),
            }
        ]
        result = make_constrained_topic_retrieval_result(
            mapped,
            {"scores": [{"chunk_id": "description-nine", "distance": 0.3}]},
        )
        self.assertEqual(result.status, "scored")
        self.assertEqual(result.scored_candidates, result.candidates)
        self.assertEqual(result.scored_candidates[0]["partition"]["year"], 3)
        self.assertEqual(
            result.scored_candidates[0]["description_evidence"][0]["provenance"],
            ({"source_page": 9},),
        )

    def test_constrained_retrieval_result_is_immutable_and_has_only_defined_states(self):
        self.assertEqual(
            CONSTRAINED_RETRIEVAL_STATES,
            (
                "empty_structural_candidates",
                "description_missing",
                "vector_missing_or_invalid",
                "no_threshold_matches",
                "scored",
            ),
        )
        result = ConstrainedTopicRetrievalResult("scored")
        with self.assertRaises(AttributeError):
            result.status = "description_missing"

    def test_constrained_topic_retrieval_embeds_once_and_scores_unique_description_ids(self):
        directory, database_path = self._course_chunk_database(
            [
                (
                    "description-one",
                    {
                        "chunk_type": "description",
                        "course_id": 1,
                        "text": "one",
                        "provenance": [{"source_page": 1}],
                    },
                )
            ]
        )
        candidates = [
            {"course_id": 1, "partition": {"plan": "coop"}},
            {"course_id": 1, "partition": {"plan": "no_coop"}},
        ]
        embedding = np.zeros(384, dtype=np.float32)
        try:
            with patch(
                "rag.retrieval.retrieve.embed_texts", return_value=np.array([embedding])
            ) as embed_mock, patch(
                "rag.retrieval.retrieve.score_candidate_vectors",
                return_value={
                    "scores": [{"chunk_id": "description-one", "distance": 0.4}],
                    "missing_chunk_ids": [],
                    "invalid_chunk_ids": [],
                },
            ) as score_mock:
                result = retrieve_constrained_topic_evidence(
                    database_path, "network", candidates
                )

            embed_mock.assert_called_once_with(["network"])
            score_mock.assert_called_once()
            self.assertEqual(score_mock.call_args.args[0], database_path)
            np.testing.assert_array_equal(score_mock.call_args.args[1], embedding)
            self.assertEqual(score_mock.call_args.args[2], ["description-one"])
            self.assertEqual(result.status, "scored")
            self.assertEqual(len(result.candidates), 2)
            self.assertEqual(
                [candidate["partition"]["plan"] for candidate in result.candidates],
                ["coop", "no_coop"],
            )
            self.assertEqual(
                [
                    candidate["description_evidence"][0]["distance"]
                    for candidate in result.candidates
                ],
                [0.4, 0.4],
            )
        finally:
            directory.cleanup()

    def test_constrained_topic_retrieval_empty_candidates_does_not_embed_or_score(self):
        with patch("rag.retrieval.retrieve.embed_texts") as embed_mock, patch(
            "rag.retrieval.retrieve.score_candidate_vectors"
        ) as score_mock:
            result = retrieve_constrained_topic_evidence(
                "missing.db", "network", []
            )

        self.assertEqual(result.status, "empty_structural_candidates")
        embed_mock.assert_not_called()
        score_mock.assert_not_called()

    def test_constrained_topic_retrieval_missing_descriptions_does_not_embed_or_score(self):
        directory, database_path = self._course_chunk_database(
            [("metadata-one", {"chunk_type": "metadata", "course_id": 1})]
        )
        try:
            with patch("rag.retrieval.retrieve.embed_texts") as embed_mock, patch(
                "rag.retrieval.retrieve.score_candidate_vectors"
            ) as score_mock:
                result = retrieve_constrained_topic_evidence(
                    database_path,
                    "network",
                    [{"course_id": 1, "partition": {"year": 2}}],
                )

            self.assertEqual(result.status, "description_missing")
            self.assertEqual(result.missing_description_course_ids, (1,))
            embed_mock.assert_not_called()
            score_mock.assert_not_called()
        finally:
            directory.cleanup()

    def test_constrained_topic_retrieval_preserves_missing_and_invalid_vector_state(self):
        directory, database_path = self._course_chunk_database(
            [
                (
                    "description-one",
                    {
                        "chunk_type": "description",
                        "course_id": 1,
                        "text": "one",
                        "provenance": [{"source_page": 1}],
                    },
                )
            ]
        )
        embedding = np.zeros(384, dtype=np.float32)
        try:
            with patch(
                "rag.retrieval.retrieve.embed_texts", return_value=np.array([embedding])
            ), patch(
                "rag.retrieval.retrieve.score_candidate_vectors",
                return_value={
                    "scores": [],
                    "missing_chunk_ids": ["description-one"],
                    "invalid_chunk_ids": ["other-invalid"],
                },
            ):
                result = retrieve_constrained_topic_evidence(
                    database_path, "network", [{"course_id": 1}]
                )

            self.assertEqual(result.status, "vector_missing_or_invalid")
            self.assertEqual(result.missing_vector_chunk_ids, ("description-one",))
            self.assertEqual(result.invalid_vector_chunk_ids, ("other-invalid",))
            self.assertIsNone(
                result.candidates[0]["description_evidence"][0]["distance"]
            )
        finally:
            directory.cleanup()

    def test_constrained_topic_threshold_is_inclusive(self):
        candidate = {
            "course_id": 1,
            "description_evidence": (
                {"chunk_id": "boundary", "distance": CONSTRAINED_TOPIC_DISTANCE_THRESHOLD},
            ),
        }
        result = make_constrained_topic_retrieval_result(
            [candidate],
            {"scores": [{"chunk_id": "boundary", "distance": CONSTRAINED_TOPIC_DISTANCE_THRESHOLD}]},
        )
        self.assertEqual(result.status, "scored")
        self.assertEqual(result.scored_candidates, (candidate,))

    def test_constrained_topic_semantic_only_match_is_accepted(self):
        candidate = {
            "course_id": 1,
            "description_evidence": (
                {"chunk_id": "semantic", "text": "course content", "distance": 0.2},
            ),
        }
        result = make_constrained_topic_retrieval_result(
            [candidate],
            {"scores": [{"chunk_id": "semantic", "distance": 0.2}]},
        )
        self.assertEqual(result.status, "scored")
        self.assertEqual(result.scored_candidates, (candidate,))

    def test_lexical_topic_match_uses_conservative_whole_phrases_and_ai_alias(self):
        self.assertTrue(
            lexical_topic_match(
                "database",
                {"name_en": "DATABASE SYSTEMS", "description_evidence": ()},
            )
        )
        self.assertFalse(
            lexical_topic_match(
                "data",
                {"name_en": "DATABASE SYSTEMS", "description_evidence": ()},
            )
        )
        self.assertTrue(
            lexical_topic_match(
                "AI",
                {"name_en": "ARTIFICIAL INTELLIGENCE", "description_evidence": ()},
            )
        )
        self.assertTrue(
            lexical_topic_match(
                "artificial intelligence",
                {"name_en": "AI", "description_evidence": ()},
            )
        )
        self.assertFalse(
            lexical_topic_match(
                "programming",
                {"name_en": "PROGRAMMERING", "description_evidence": ()},
            )
        )
        self.assertFalse(
            lexical_topic_match(
                "database",
                {
                    "name_en": "DATA MINING AND ANALYTICS",
                    "description_evidence": (),
                },
            )
        )

    def test_lexical_only_rescue_accepts_above_threshold_candidate(self):
        directory, database_path = self._course_chunk_database(
            [
                (
                    "description-one",
                    {
                        "chunk_type": "description",
                        "course_id": 1,
                        "text": "course content",
                        "provenance": [{"source_page": 1}],
                    },
                )
            ]
        )
        try:
            with patch(
                "rag.retrieval.retrieve.embed_texts",
                return_value=np.ones((1, 384), dtype=np.float32),
            ), patch(
                "rag.retrieval.retrieve.score_candidate_vectors",
                return_value={
                    "scores": [{"chunk_id": "description-one", "distance": 0.9}],
                    "missing_chunk_ids": [],
                    "invalid_chunk_ids": [],
                },
            ):
                result = retrieve_constrained_topic_evidence(
                    database_path,
                    "AI",
                    [
                        {
                            "course_id": 1,
                            "name_en": "ARTIFICIAL INTELLIGENCE",
                            "partition": {"plan": "coop"},
                        }
                    ],
                )
            self.assertEqual(result.status, "scored")
            self.assertEqual(len(result.scored_candidates), 1)
            self.assertEqual(
                result.scored_candidates[0]["description_evidence"][0]["distance"],
                0.9,
            )
        finally:
            directory.cleanup()

    def test_constrained_topic_match_reports_no_threshold_matches(self):
        candidate = {
            "course_id": 1,
            "description_evidence": (
                {"chunk_id": "far", "text": "unrelated", "distance": 0.8},
            ),
        }
        result = make_constrained_topic_retrieval_result(
            [candidate],
            {"scores": [{"chunk_id": "far", "distance": 0.8}]},
            (),
        )
        self.assertEqual(result.status, "no_threshold_matches")
        self.assertEqual(result.scored_candidates, ())
        self.assertEqual(result.candidates[0]["description_evidence"][0]["distance"], 0.8)

    def test_lexical_rescue_accepts_missing_or_invalid_vector(self):
        candidate = {
            "course_id": 1,
            "name_en": "COMPUTER NETWORKS",
            "description_evidence": (
                {"chunk_id": "missing-vector", "text": "course content", "distance": None},
            ),
        }
        result = make_constrained_topic_retrieval_result(
            [candidate],
            {
                "scores": [],
                "missing_chunk_ids": ["missing-vector"],
                "invalid_chunk_ids": [],
            },
            (candidate,),
        )
        self.assertEqual(result.status, "scored")
        self.assertEqual(result.scored_candidates, (candidate,))

    def test_lexical_topic_match_preserves_partition_and_provenance_context(self):
        provenance = {"source_page": 7}
        candidate = {
            "course_id": 1,
            "name_en": "NETWORK SYSTEMS",
            "partition": {"plan": "coop", "year": 2},
            "description_evidence": (
                {
                    "chunk_id": "network-description",
                    "text": "network content",
                    "distance": 0.8,
                    "provenance": [provenance],
                },
            ),
        }
        self.assertTrue(lexical_topic_match("network", candidate))
        result = make_constrained_topic_retrieval_result(
            [candidate],
            {"scores": [{"chunk_id": "network-description", "distance": 0.8}]},
            (candidate,),
        )
        self.assertEqual(result.scored_candidates[0]["partition"], candidate["partition"])
        self.assertEqual(
            result.scored_candidates[0]["description_evidence"][0]["provenance"],
            (provenance,),
        )


if __name__ == "__main__":
    unittest.main()
