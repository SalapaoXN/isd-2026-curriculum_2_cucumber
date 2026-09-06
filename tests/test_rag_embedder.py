import unittest
from unittest.mock import patch

import numpy as np
import torch

from rag.retrieval import embedder


class _Tokenizer:
    def __init__(self):
        self.batches = []

    def __call__(self, texts, **kwargs):
        self.batches.append(list(texts))
        return {
            "attention_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.int64)
        }


class _Model:
    def eval(self):
        return self

    def __call__(self, **kwargs):
        return (
            torch.tensor(
                [
                    [[1.0, 0.0], [3.0, 0.0]],
                    [[0.0, 2.0], [0.0, 4.0]],
                ]
            ),
        )


class RagEmbedderTest(unittest.TestCase):
    def test_lazy_ordered_float32_embeddings(self):
        tokenizer = _Tokenizer()
        model = _Model()
        embedder._COMPONENTS = None

        with patch.object(
            embedder,
            "_load_components",
            return_value=(tokenizer, model, torch),
        ) as load_components:
            empty = embedder.embed_texts([])
            self.assertEqual(empty.shape, (0, embedder.EMBEDDING_DIMENSION))
            load_components.assert_not_called()

            result = embedder.embed_texts(["first", "second"])
            again = embedder.embed_texts(["first", "second"])

        load_components.assert_called_once_with()
        self.assertEqual(tokenizer.batches, [["first", "second"], ["first", "second"]])
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [[2.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_equal(result, again)


if __name__ == "__main__":
    unittest.main()
