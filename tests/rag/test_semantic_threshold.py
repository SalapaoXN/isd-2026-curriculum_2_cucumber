import unittest

from rag.retrieval.threshold import (
    FROZEN_EMBEDDING_DIMENSION,
    FROZEN_EMBEDDING_MODEL,
    ThresholdCalibrationResult,
    calibrate_threshold,
    is_topic_match,
)


class SemanticThresholdTests(unittest.TestCase):
    def _sample(self, sample_id, distance, relevant, **overrides):
        sample = {
            "id": sample_id,
            "topic": "network",
            "course_id": sample_id,
            "chunk_id": f"description-{sample_id}",
            "distance": distance,
            "relevant": relevant,
            "embedding_model": FROZEN_EMBEDDING_MODEL,
            "embedding_dimension": FROZEN_EMBEDDING_DIMENSION,
        }
        sample.update(overrides)
        return sample

    def test_calibration_selects_maximum_f1_deterministically(self):
        samples = [
            self._sample("a", 0.20, True),
            self._sample("b", 0.40, True),
            self._sample("c", 0.50, False),
            self._sample("d", 0.80, False),
        ]
        first = calibrate_threshold(samples)
        second = calibrate_threshold(reversed(samples))

        self.assertIsInstance(first, ThresholdCalibrationResult)
        self.assertEqual(first, second)
        self.assertEqual(first.threshold, 0.4)
        self.assertEqual(first.f1, 1.0)

    def test_prediction_includes_equal_boundary(self):
        self.assertTrue(is_topic_match(0.4, 0.4))
        self.assertFalse(is_topic_match(0.400001, 0.4))

    def test_ties_prefer_higher_precision_then_lower_threshold(self):
        samples = [
            self._sample("relevant", 0.20, True),
            self._sample("near-negative", 0.30, False),
            self._sample("far-negative", 0.40, False),
            self._sample("later-relevant", 0.50, True),
        ]
        result = calibrate_threshold(samples)
        self.assertEqual(result.threshold, 0.2)
        self.assertEqual(result.precision, 1.0)

        no_positive_samples = [
            self._sample("first-negative", 0.20, False),
            self._sample("second-negative", 0.40, False),
        ]
        self.assertEqual(calibrate_threshold(no_positive_samples).threshold, 0.2)

    def test_model_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_threshold(
                [self._sample("a", 0.2, True, embedding_model="other-model")]
            )

    def test_dimension_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_threshold(
                [self._sample("a", 0.2, True, embedding_dimension=768)]
            )

    def test_malformed_samples_are_rejected(self):
        malformed = self._sample("a", float("nan"), True)
        with self.assertRaises(ValueError):
            calibrate_threshold([malformed])

        malformed = self._sample("a", 0.2, 1)
        with self.assertRaises(ValueError):
            calibrate_threshold([malformed])

        malformed = self._sample("a", 0.2, True, course_id=None, chunk_id=None)
        with self.assertRaises(ValueError):
            calibrate_threshold([malformed])

    def test_empty_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_threshold([])


if __name__ == "__main__":
    unittest.main()
