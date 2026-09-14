"""Pure calibration utilities for the constrained semantic threshold."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

from .embedder import EMBEDDING_DIMENSION, MODEL_NAME


FROZEN_EMBEDDING_MODEL = MODEL_NAME
FROZEN_EMBEDDING_DIMENSION = EMBEDDING_DIMENSION


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationResult:
    """Deterministic result of calibrating one fixed threshold."""

    threshold: float
    f1: float
    precision: float
    recall: float
    sample_count: int
    embedding_model: str
    embedding_dimension: int


@dataclass(frozen=True, slots=True)
class _ValidatedSample:
    distance: float
    relevant: bool


def _validated_samples(
    samples: Iterable[Mapping[str, Any]],
) -> tuple[_ValidatedSample, ...]:
    validated: list[_ValidatedSample] = []
    model: str | None = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"sample {index} must be an object")
        required = {
            "id",
            "topic",
            "distance",
            "relevant",
            "embedding_model",
            "embedding_dimension",
        }
        missing = required - set(sample)
        if missing:
            raise ValueError(f"sample {index} missing fields: {sorted(missing)!r}")

        sample_id = sample["id"]
        topic = sample["topic"]
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"sample {index} id must be non-empty")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError(f"sample {index} topic must be non-empty")

        has_course_id = "course_id" in sample and sample["course_id"] not in (None, "")
        has_chunk_id = "chunk_id" in sample and sample["chunk_id"] not in (None, "")
        if not has_course_id and not has_chunk_id:
            raise ValueError(f"sample {index} needs course_id or chunk_id")
        if has_chunk_id and not isinstance(sample["chunk_id"], str):
            raise ValueError(f"sample {index} chunk_id must be a string")

        distance = sample["distance"]
        if isinstance(distance, bool) or not isinstance(distance, Real):
            raise ValueError(f"sample {index} distance must be numeric")
        distance = float(distance)
        if not math.isfinite(distance) or not 0.0 <= distance <= 2.0:
            raise ValueError(f"sample {index} distance must be finite in [0, 2]")
        if not isinstance(sample["relevant"], bool):
            raise ValueError(f"sample {index} relevant must be boolean")

        sample_model = sample["embedding_model"]
        sample_dimension = sample["embedding_dimension"]
        if not isinstance(sample_model, str) or not sample_model.strip():
            raise ValueError(f"sample {index} embedding_model must be non-empty")
        if sample_model != FROZEN_EMBEDDING_MODEL:
            raise ValueError(
                f"sample {index} uses unsupported embedding model: {sample_model!r}"
            )
        if (
            isinstance(sample_dimension, bool)
            or not isinstance(sample_dimension, int)
            or sample_dimension != FROZEN_EMBEDDING_DIMENSION
        ):
            raise ValueError(
                f"sample {index} embedding_dimension must be "
                f"{FROZEN_EMBEDDING_DIMENSION}"
            )
        if model is None:
            model = sample_model
        elif sample_model != model:
            raise ValueError("all calibration samples must use one embedding model")

        validated.append(_ValidatedSample(distance, sample["relevant"]))

    if not validated:
        raise ValueError("at least one calibration sample is required")
    return tuple(validated)


def _metrics(
    samples: tuple[_ValidatedSample, ...],
    threshold: float,
) -> tuple[float, float, float]:
    true_positive = sum(
        sample.relevant and sample.distance <= threshold for sample in samples
    )
    false_positive = sum(
        not sample.relevant and sample.distance <= threshold for sample in samples
    )
    false_negative = sum(
        sample.relevant and sample.distance > threshold for sample in samples
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return f1, precision, recall


def calibrate_threshold(
    samples: Iterable[Mapping[str, Any]],
) -> ThresholdCalibrationResult:
    """Select one threshold from observed distances using deterministic F1."""
    validated = _validated_samples(samples)
    thresholds = sorted({sample.distance for sample in validated})
    candidates = []
    for threshold in thresholds:
        f1, precision, recall = _metrics(validated, threshold)
        candidates.append((f1, precision, -threshold, threshold, recall))
    _, _, _, threshold, recall = max(candidates)
    f1, precision, recall = _metrics(validated, threshold)
    return ThresholdCalibrationResult(
        threshold=threshold,
        f1=f1,
        precision=precision,
        recall=recall,
        sample_count=len(validated),
        embedding_model=FROZEN_EMBEDDING_MODEL,
        embedding_dimension=FROZEN_EMBEDDING_DIMENSION,
    )


def is_topic_match(distance: Real, threshold: Real) -> bool:
    """Apply the fixed inclusive distance boundary without dynamic behavior."""
    if isinstance(distance, bool) or not isinstance(distance, Real):
        raise ValueError("distance must be numeric")
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise ValueError("threshold must be numeric")
    distance = float(distance)
    threshold = float(threshold)
    if not math.isfinite(distance) or not math.isfinite(threshold):
        raise ValueError("distance and threshold must be finite")
    return distance <= threshold


__all__ = [
    "FROZEN_EMBEDDING_DIMENSION",
    "FROZEN_EMBEDDING_MODEL",
    "ThresholdCalibrationResult",
    "calibrate_threshold",
    "is_topic_match",
]
