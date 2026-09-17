import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

import numpy as np

from rag.retrieval.retrieve import (
    SIMILARITY_STATES,
    SimilarityEvidence,
    aggregate_exact_course_similarity,
    compare_exact_description_vectors,
)
from rag.retrieval.vector_store import (
    compare_stored_vectors,
    create_vector_table,
    insert_embeddings,
)


class RagSimilarityTest(unittest.TestCase):
    def _database(self, descriptions, vectors):
        directory = tempfile.TemporaryDirectory()
        path = f"{directory.name}/curriculum.db"
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("CREATE TABLE semantic_chunks (chunk_id TEXT, chunk_json TEXT)")
            connection.executemany(
                "INSERT INTO semantic_chunks VALUES (?, ?)",
                [(chunk_id, json.dumps(chunk)) for chunk_id, chunk in descriptions],
            )
            connection.execute("CREATE TABLE vector (chunk_id TEXT, embed_byte BLOB)")
            connection.executemany(
                "INSERT INTO vector VALUES (?, ?)",
                [
                    (chunk_id, np.asarray(vector, dtype=np.float32).tobytes())
                    for chunk_id, vector in vectors
                ],
            )
            connection.commit()
        return directory, path

    def _description(self, chunk_id, code, plan, text, page, program="IT"):
        return (
            chunk_id,
            {
                "chunk_id": chunk_id,
                "chunk_type": "description",
                "course_id": code,
                "course_code": code,
                "program": program,
                "plan": plan,
                "text": text,
                "provenance": [{"source_page": page}],
            },
        )

    def _evidence(self, chunk_id, code, plan, text, page, program="IT"):
        return {
            "chunk_id": chunk_id,
            "chunk_type": "description",
            "course_id": code,
            "course_code": code,
            "program": program,
            "text": text,
            "provenance": ({"source_page": page},),
            "partition": {"plan": plan},
        }

    def _course(self, code, plan, descriptions, program="IT"):
        return {
            "program": program,
            "course_code": code,
            "partition": {"plan": plan},
            "description_evidence": tuple(descriptions),
        }

    def _valid_pair_database(self):
        return self._database(
            [
                self._description("left", "00000001", "coop", "left text", 1),
                self._description("right", "00000002", "coop", "right text", 2),
            ],
            [
                ("left", [1] + [0] * 383),
                ("right", [1] + [0] * 383),
            ],
        )

    def test_single_matching_partition_returns_raw_pair(self):
        directory, path = self._database(
            [
                self._description("left", "00000001", "coop", "left text", 1),
                self._description("right", "00000002", "coop", "right text", 2),
            ],
            [("left", [1] + [0] * 383), ("right", [1] + [0] * 383)],
        )
        try:
            result = compare_exact_description_vectors(path, "left", "right")
        finally:
            directory.cleanup()

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.cosine_distance, 0.0)
        self.assertEqual(result.cosine_similarity, 1.0)
        self.assertEqual(result.left["chunk_id"], "left")
        self.assertEqual(result.right["chunk_id"], "right")
        self.assertEqual(result.left["provenance"], ({"source_page": 1},))

    def test_invalid_public_chunk_ids_fail_closed(self):
        directory, path = self._valid_pair_database()
        try:
            for left_chunk_id in (None, "", "   "):
                with self.subTest(left_chunk_id=left_chunk_id):
                    result = compare_exact_description_vectors(
                        path, left_chunk_id, "right"
                    )
                    self.assertEqual(result.status, "insufficient_evidence")
                    self.assertEqual(result.reason, "invalid_supplied_evidence")
                    self.assertEqual(result.left, {})
                    self.assertEqual(result.right, {})
        finally:
            directory.cleanup()

    def test_invalid_selected_plan_fails_closed(self):
        directory, path = self._valid_pair_database()
        try:
            result = aggregate_exact_course_similarity(
                path,
                self._course("00000001", "coop", ()),
                self._course("00000002", "coop", ()),
                selected_plan="",
            )
        finally:
            directory.cleanup()

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.pairs, ())

    def test_coop_and_no_coop_are_compared_only_with_aligned_partitions(self):
        descriptions = [
            self._description("lc", "00000001", "coop", "coop left", 1),
            self._description("rc", "00000002", "coop", "coop right", 2),
            self._description("ln", "00000001", "no_coop", "no left", 3),
            self._description("rn", "00000002", "no_coop", "no right", 4),
        ]
        vectors = [
            ("lc", [1] + [0] * 383),
            ("rc", [1] + [0] * 383),
            ("ln", [1] + [0] * 383),
            ("rn", [0, 1] + [0] * 382),
        ]
        directory, path = self._database(descriptions, vectors)
        try:
            result = aggregate_exact_course_similarity(
                path,
                [self._course("00000001", "coop", [self._evidence("lc", "00000001", "coop", "coop left", 1)]),
                 self._course("00000001", "no_coop", [self._evidence("ln", "00000001", "no_coop", "no left", 3)])],
                [self._course("00000002", "coop", [self._evidence("rc", "00000002", "coop", "coop right", 2)]),
                 self._course("00000002", "no_coop", [self._evidence("rn", "00000002", "no_coop", "no right", 4)])],
            )
        finally:
            directory.cleanup()

        self.assertEqual(result.status, "complete")
        self.assertEqual([pair.partition["plan"] for pair in result.pairs], ["coop", "no_coop"])
        self.assertEqual([pair.cosine_distance for pair in result.pairs], [0.0, 1.0])

    def test_cross_program_same_plan_pair_is_compared_without_merging_identity(self):
        directory, path = self._database(
            [
                self._description("left", "00000001", "coop", "left", 1, "IT"),
                self._description("right", "00000002", "coop", "right", 2, "DSBA"),
            ],
            [("left", [1] + [0] * 383), ("right", [1] + [0] * 383)],
        )
        try:
            result = aggregate_exact_course_similarity(
                path,
                self._course(
                    "00000001",
                    "coop",
                    [self._evidence("left", "00000001", "coop", "left", 1, "IT")],
                    "IT",
                ),
                self._course(
                    "00000002",
                    "coop",
                    [self._evidence("right", "00000002", "coop", "right", 2, "DSBA")],
                    "DSBA",
                ),
            )
        finally:
            directory.cleanup()

        self.assertEqual(result.status, "complete")
        self.assertEqual(len(result.pairs), 1)
        self.assertEqual(result.pairs[0].partition["plan"], "coop")
        self.assertEqual(result.pairs[0].left["program"], "IT")
        self.assertEqual(result.pairs[0].right["program"], "DSBA")

    def test_nonmatching_partitions_are_not_cross_compared(self):
        directory, path = self._database([], [])
        try:
            result = aggregate_exact_course_similarity(
                path,
                self._course("00000001", "coop", [self._evidence("left", "00000001", "coop", "left", 1)]),
                self._course("00000002", "no_coop", [self._evidence("right", "00000002", "no_coop", "right", 2)]),
            )
        finally:
            directory.cleanup()

        self.assertEqual(result.status, "valid_empty")
        self.assertEqual(result.pairs, ())
        self.assertEqual(
            {partition["plan"] for partition in result.unmatched_partitions},
            {"coop", "no_coop"},
        )

    def test_selected_plan_limits_comparison(self):
        descriptions = [
            self._description("lc", "00000001", "coop", "coop left", 1),
            self._description("rc", "00000002", "coop", "coop right", 2),
            self._description("ln", "00000001", "no_coop", "no left", 3),
            self._description("rn", "00000002", "no_coop", "no right", 4),
        ]
        vectors = [(chunk_id, [1] + [0] * 383) for chunk_id, _ in descriptions]
        directory, path = self._database(descriptions, vectors)
        try:
            result = aggregate_exact_course_similarity(
                path,
                [self._course("00000001", "coop", [self._evidence("lc", "00000001", "coop", "coop left", 1)]),
                 self._course("00000001", "no_coop", [self._evidence("ln", "00000001", "no_coop", "no left", 3)])],
                [self._course("00000002", "coop", [self._evidence("rc", "00000002", "coop", "coop right", 2)]),
                 self._course("00000002", "no_coop", [self._evidence("rn", "00000002", "no_coop", "no right", 4)])],
                selected_plan="coop",
            )
        finally:
            directory.cleanup()

        self.assertEqual(result.status, "complete")
        self.assertEqual(len(result.pairs), 1)
        self.assertEqual(result.pairs[0].partition["plan"], "coop")

    def test_multiple_common_partitions_have_deterministic_summaries(self):
        descriptions = [
            self._description("l1", "00000001", "coop", "one left", 1),
            self._description("r1", "00000002", "coop", "one right", 2),
            self._description("l2", "00000001", "no_coop", "two left", 3),
            self._description("r2", "00000002", "no_coop", "two right", 4),
        ]
        vectors = [
            ("l1", [1] + [0] * 383),
            ("r1", [1] + [0] * 383),
            ("l2", [1] + [0] * 383),
            ("r2", [0, 1] + [0] * 382),
        ]
        directory, path = self._database(descriptions, vectors)
        try:
            result = aggregate_exact_course_similarity(
                path,
                [self._course("00000001", "coop", [self._evidence("l1", "00000001", "coop", "one left", 1)]),
                 self._course("00000001", "no_coop", [self._evidence("l2", "00000001", "no_coop", "two left", 3)])],
                [self._course("00000002", "coop", [self._evidence("r1", "00000002", "coop", "one right", 2)]),
                 self._course("00000002", "no_coop", [self._evidence("r2", "00000002", "no_coop", "two right", 4)])],
            )
        finally:
            directory.cleanup()

        self.assertEqual(result.mean_distance, 0.5)
        self.assertEqual(result.min_distance, 0.0)
        self.assertEqual(result.max_distance, 1.0)
        self.assertFalse(hasattr(result, "similar"))

    def test_missing_description_is_insufficient_for_matching_partition(self):
        directory, path = self._database([], [])
        try:
            result = aggregate_exact_course_similarity(
                path,
                self._course("00000001", "coop", ()),
                self._course("00000002", "coop", ()),
            )
        finally:
            directory.cleanup()

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.pairs[0].reason, "description_missing")

    def test_missing_and_invalid_vectors_are_explicit(self):
        descriptions = [
            self._description("left", "00000001", "coop", "left", 1),
            self._description("missing", "00000002", "coop", "missing", 2),
            self._description("invalid", "00000003", "coop", "invalid", 3),
        ]
        directory, path = self._database(
            descriptions,
            [("left", [1] + [0] * 383), ("invalid", [0] * 384)],
        )
        try:
            missing = compare_exact_description_vectors(path, "left", "missing")
            invalid = compare_exact_description_vectors(path, "left", "invalid")
        finally:
            directory.cleanup()

        self.assertEqual(missing.status, "insufficient_evidence")
        self.assertEqual(missing.missing_chunk_ids, ("missing",))
        self.assertEqual(invalid.status, "insufficient_evidence")
        self.assertEqual(invalid.invalid_chunk_ids, ("invalid",))

    def test_metadata_supplied_evidence_is_rejected(self):
        directory, path = self._valid_pair_database()
        try:
            left = self._evidence("left", "00000001", "coop", "left text", 1)
            left["chunk_type"] = "metadata"
            result = compare_exact_description_vectors(
                path,
                "left",
                "right",
                left_evidence=left,
            )
        finally:
            directory.cleanup()

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.reason, "invalid_supplied_evidence")

    def test_chunk_course_identity_mismatch_is_rejected(self):
        directory, path = self._valid_pair_database()
        try:
            left = self._evidence("left", "00000099", "coop", "left text", 1)
            result = compare_exact_description_vectors(
                path,
                "left",
                "right",
                left_evidence=left,
            )
        finally:
            directory.cleanup()

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.reason, "inconsistent_evidence")

    def test_missing_text_or_provenance_is_rejected(self):
        directory, path = self._valid_pair_database()
        try:
            missing_text = self._evidence("left", "00000001", "coop", "left text", 1)
            missing_text.pop("text")
            missing_provenance = self._evidence(
                "left", "00000001", "coop", "left text", 1
            )
            missing_provenance.pop("provenance")
            text_result = compare_exact_description_vectors(
                path,
                "left",
                "right",
                left_evidence=missing_text,
            )
            provenance_result = compare_exact_description_vectors(
                path,
                "left",
                "right",
                left_evidence=missing_provenance,
            )
        finally:
            directory.cleanup()

        self.assertEqual(text_result.reason, "invalid_supplied_evidence")
        self.assertEqual(provenance_result.reason, "invalid_supplied_evidence")

    def test_missing_or_empty_partition_is_rejected(self):
        directory, path = self._valid_pair_database()
        try:
            missing = self._evidence("left", "00000001", "coop", "left text", 1)
            missing.pop("partition")
            empty = self._evidence("left", "00000001", "coop", "left text", 1)
            empty["partition"] = {}
            missing_result = compare_exact_description_vectors(
                path,
                "left",
                "right",
                left_evidence=missing,
            )
            empty_result = compare_exact_description_vectors(
                path,
                "left",
                "right",
                left_evidence=empty,
            )
        finally:
            directory.cleanup()

        self.assertEqual(missing_result.reason, "invalid_supplied_evidence")
        self.assertEqual(empty_result.reason, "invalid_supplied_evidence")

    def test_explicit_partition_inconsistency_is_rejected(self):
        directory, path = self._valid_pair_database()
        try:
            result = compare_exact_description_vectors(
                path,
                "left",
                "right",
                left_evidence=self._evidence("left", "00000001", "coop", "left text", 1),
                right_evidence=self._evidence("right", "00000002", "coop", "right text", 2),
                partition={"plan": "no_coop"},
            )
        finally:
            directory.cleanup()

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.reason, "partition_mismatch")

    def test_malformed_supplied_evidence_fails_closed_without_exception(self):
        directory, path = self._valid_pair_database()
        try:
            result = compare_exact_description_vectors(
                path,
                "left",
                "right",
                left_evidence={"chunk_id": "left"},
            )
        finally:
            directory.cleanup()

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.reason, "invalid_supplied_evidence")

    def test_persisted_vector_alone_cannot_validate_fabricated_evidence(self):
        directory, path = self._valid_pair_database()
        try:
            fabricated = self._evidence(
                "left", "00000001", "coop", "fabricated text", 1
            )
            result = compare_exact_description_vectors(
                path,
                "left",
                "right",
                left_evidence=fabricated,
            )
        finally:
            directory.cleanup()

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.reason, "inconsistent_evidence")

    def test_non_identical_descriptions_in_one_partition_are_insufficient(self):
        directory, path = self._database([], [])
        try:
            result = aggregate_exact_course_similarity(
                path,
                self._course(
                    "00000001",
                    "coop",
                    [
                        self._evidence("left-a", "00000001", "coop", "first", 1),
                        self._evidence("left-b", "00000001", "coop", "second", 2),
                    ],
                ),
                self._course("00000002", "coop", [self._evidence("right", "00000002", "coop", "right", 3)]),
            )
        finally:
            directory.cleanup()

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.pairs[0].reason, "multiple_non_identical_descriptions")

    def test_pairwise_vector_helper_is_not_global_search(self):
        directory, path = self._database(
            [],
            [("left", [1] + [0] * 383), ("right", [0, 1] + [0] * 382)],
        )
        try:
            with patch("rag.retrieval.vector_store.nearest_neighbor_search") as search:
                result = compare_stored_vectors(path, "left", "right")
        finally:
            directory.cleanup()

        search.assert_not_called()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["cosine_distance"], 1.0)
        self.assertEqual(result["cosine_similarity"], 0.0)

    def test_exact_similarity_works_with_real_vec0_virtual_table(self):
        directory = tempfile.TemporaryDirectory()
        path = f"{directory.name}/curriculum.db"
        left = self._evidence("left", "00000001", "coop", "left text", 1)
        right = self._evidence("right", "00000002", "coop", "right text", 2)
        descriptions = [
            self._description("left", "00000001", "coop", "left text", 1),
            self._description("right", "00000002", "coop", "right text", 2),
        ]
        try:
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    "CREATE TABLE semantic_chunks (chunk_id TEXT, chunk_json TEXT)"
                )
                connection.executemany(
                    "INSERT INTO semantic_chunks VALUES (?, ?)",
                    [(chunk_id, json.dumps(chunk)) for chunk_id, chunk in descriptions],
                )
                connection.commit()
            create_vector_table(path)
            vector = np.zeros(384, dtype=np.float32)
            vector[0] = 1.0
            insert_embeddings(
                path,
                [{"chunk_id": "left"}, {"chunk_id": "right"}],
                [vector, vector],
            )

            result = compare_exact_description_vectors(
                path,
                "left",
                "right",
                left_evidence=left,
                right_evidence=right,
                partition={"plan": "coop"},
            )
        finally:
            directory.cleanup()

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.cosine_distance, 0.0)
        self.assertEqual(result.cosine_similarity, 1.0)

    def test_input_evidence_and_states_are_unchanged(self):
        left = self._evidence("left", "00000001", "coop", "left", 1)
        right = self._evidence("right", "00000002", "coop", "right", 2)
        left_before = dict(left)
        right_before = dict(right)
        directory, path = self._database(
            [
                self._description("left", "00000001", "coop", "left", 1),
                self._description("right", "00000002", "coop", "right", 2),
            ],
            [("left", [1] + [0] * 383), ("right", [1] + [0] * 383)],
        )
        try:
            result = aggregate_exact_course_similarity(
                path,
                self._course("00000001", "coop", [left]),
                self._course("00000002", "coop", [right]),
            )
        finally:
            directory.cleanup()

        self.assertEqual(left, left_before)
        self.assertEqual(right, right_before)
        self.assertEqual(SIMILARITY_STATES, ("complete", "valid_empty", "insufficient_evidence"))
        self.assertIsInstance(result, SimilarityEvidence)
        with self.assertRaises(AttributeError):
            result.status = "valid_empty"


if __name__ == "__main__":
    unittest.main()
