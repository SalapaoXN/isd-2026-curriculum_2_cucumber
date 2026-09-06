import unittest
from unittest.mock import patch

import numpy as np

from rag.retrieval.retrieve import retrieve


class RagRetrieveTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
