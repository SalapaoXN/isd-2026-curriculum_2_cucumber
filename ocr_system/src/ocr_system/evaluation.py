import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any
from jiwer import wer
import Levenshtein


@dataclass
class EvaluationResult:
    file: str
    cer: float
    wer: float
    exact_match: bool
    reference_chars: int
    prediction_chars: int


@dataclass
class ExtractedEvaluationResult:
    file: str
    exact_match: bool
    reference_courses: int
    prediction_courses: int
    matched_courses: int
    precision: float
    recall: float
    f1: float


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def char_error_rate(reference: str, prediction: str) -> float:
    ref = normalize_text(reference).replace(" ", "")
    hyp = normalize_text(prediction).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def evaluate_text(reference: str, prediction: str, file_name: str = "") -> EvaluationResult:
    ref = normalize_text(reference)
    hyp = normalize_text(prediction)
    return EvaluationResult(
        file=file_name,
        cer=char_error_rate(ref, hyp),
        wer=wer(ref, hyp) if ref else (0.0 if not hyp else 1.0),
        exact_match=ref == hyp,
        reference_chars=len(ref),
        prediction_chars=len(hyp),
    )


def evaluate_from_files(ground_truth_json: str | Path, prediction_json: str | Path) -> dict:
    with Path(ground_truth_json).open("r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    with Path(prediction_json).open("r", encoding="utf-8") as f:
        prediction = json.load(f)

    source_name = Path(prediction["source_path"]).name
    reference = ground_truth.get(source_name) or ground_truth.get(Path(source_name).stem)
    if reference is None:
        raise KeyError(f"No ground truth found for {source_name}")

    result = evaluate_text(reference, prediction["text"], file_name=source_name)
    return asdict(result)


def evaluate_extracted_from_files(ground_truth_json: str | Path, prediction_json: str | Path) -> dict[str, Any]:
    with Path(ground_truth_json).open("r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    with Path(prediction_json).open("r", encoding="utf-8") as f:
        prediction = json.load(f)

    reference_courses = ground_truth.get("courses", [])
    prediction_courses = prediction.get("courses", [])

    reference_keys = {
        _course_key(course)
        for course in reference_courses
        if _course_key(course)
    }
    prediction_keys = {
        _course_key(course)
        for course in prediction_courses
        if _course_key(course)
    }

    matched_courses = len(reference_keys & prediction_keys)
    precision = matched_courses / len(prediction_keys) if prediction_keys else 0.0
    recall = matched_courses / len(reference_keys) if reference_keys else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    result = ExtractedEvaluationResult(
        file=Path(prediction_json).name,
        exact_match=matched_courses == len(reference_keys) == len(prediction_keys),
        reference_courses=len(reference_keys),
        prediction_courses=len(prediction_keys),
        matched_courses=matched_courses,
        precision=precision,
        recall=recall,
        f1=f1,
    )
    return asdict(result)


def _course_key(course: dict[str, Any]) -> str | None:
    code = str(course.get("code") or "").strip()
    if not code:
        return None
    return code.lower()

