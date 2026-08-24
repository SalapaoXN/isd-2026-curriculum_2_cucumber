"""Opt-in English-only course-name enrichment for the production pipeline."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CODE_RE = re.compile(r"^\d{8}$")
PLACEHOLDER_CODE_RE = re.compile(r"^\d{4,8}X+$")
CODE_PREFIX_RE = re.compile(r"^(?:\d{8}|\d{4,8}X+)\s+\S")
CREDIT_RE = re.compile(r"\d+\s*\(\s*\d+\s*[-–]\s*\d+\s*[-–]\s*\d+\s*\)")
PREREQUISITE_RE = re.compile(r"^PRERE[A-Z]*\b", re.IGNORECASE)
HAS_ASCII_LETTER_RE = re.compile(r"[A-Z]", re.IGNORECASE)
PAGE_RE = re.compile(r"page_(\d+)$", re.IGNORECASE)

MIN_TITLE_CONFIDENCE = 0.5
SINGLE_BAND_SPAN_FACTOR = 1.5
PROVENANCE_KEY = "english_second_pass"


@dataclass(frozen=True)
class Detection:
    """One English OCR detail=1 detection in spatial reading order."""

    original_index: int
    bbox: list[list[float]]
    text: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def exact_numeric_code(value: Any) -> str | None:
    """Return only an exact normalized eight-digit code."""

    text = re.sub(r"\s+", "", str(value or "").strip())
    return text if CODE_RE.fullmatch(text) else None


def placeholder_code_like(value: Any) -> bool:
    text = re.sub(r"\s+", "", str(value or "").strip()).upper()
    return bool(PLACEHOLDER_CODE_RE.fullmatch(text))


def code_prefixed_row_like(value: Any) -> bool:
    text = str(value or "").strip().upper()
    return bool(CODE_PREFIX_RE.match(text))


def _row_boundary_like(value: Any) -> bool:
    return (
        exact_numeric_code(value) is not None
        or placeholder_code_like(value)
        or code_prefixed_row_like(value)
    )


def _bbox_values(
    bbox: Iterable[Iterable[Any]],
) -> tuple[list[list[float]], float, float, float, float]:
    points = [[float(point[0]), float(point[1])] for point in bbox]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return points, min(xs), min(ys), max(xs), max(ys)


def ordered_detections(results: Iterable[Any]) -> list[Detection]:
    detections: list[Detection] = []
    for original_index, result in enumerate(results):
        if not isinstance(result, (list, tuple)) or len(result) != 3:
            continue
        bbox, text, confidence = result
        try:
            points, x_min, y_min, x_max, y_max = _bbox_values(bbox)
            detections.append(
                Detection(
                    original_index=original_index,
                    bbox=points,
                    text=str(text).strip().upper(),
                    confidence=float(confidence),
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                )
            )
        except (TypeError, ValueError, IndexError):
            continue

    return sorted(
        detections,
        key=lambda item: (item.y_min, item.x_min, item.original_index),
    )


def _is_credit_detection(detection: Detection) -> bool:
    return bool(CREDIT_RE.search(detection.text))


def _is_prerequisite_detection(detection: Detection) -> bool:
    return bool(PREREQUISITE_RE.match(detection.text.strip()))


def _is_title_detection(detection: Detection) -> bool:
    text = detection.text.strip()
    if not text or _row_boundary_like(text) or _is_credit_detection(detection):
        return False
    if _is_prerequisite_detection(detection) or text == "NONE":
        return False
    if text.isdigit():
        return text in {"1", "2", "3", "4", "5", "6"}
    return bool(HAS_ASCII_LETTER_RE.search(text))


def _detection_provenance(detection: Detection) -> dict[str, Any]:
    return {
        "detection_index": detection.original_index,
        "text": detection.text,
        "confidence": detection.confidence,
        "bbox": detection.bbox,
    }


def _union_bbox(detections: list[Detection]) -> list[float]:
    return [
        min(item.x_min for item in detections),
        min(item.y_min for item in detections),
        max(item.x_max for item in detections),
        max(item.y_max for item in detections),
    ]


def _single_title_band(detections: list[Detection]) -> bool:
    if not detections:
        return False
    span = max(item.y_max for item in detections) - min(
        item.y_min for item in detections
    )
    max_height = max(item.y_max - item.y_min for item in detections)
    return span <= max_height * SINGLE_BAND_SPAN_FACTOR


def candidate_from_anchor(
    ordered: list[Detection],
    anchor_position: int,
    next_row_boundary_position: int | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Extract a conservative single-band title after a code/credit pair."""

    end = (
        next_row_boundary_position
        if next_row_boundary_position is not None
        else len(ordered)
    )
    block = ordered[anchor_position + 1 : end]
    preceding_credit = (
        ordered[anchor_position - 1]
        if anchor_position > 0 and _is_credit_detection(ordered[anchor_position - 1])
        else None
    )
    if preceding_credit is not None:
        credit_detection = preceding_credit
        after_credit = block
    else:
        credit_positions = [
            index for index, item in enumerate(block) if _is_credit_detection(item)
        ]
        if not credit_positions:
            return None, "missing_credit_anchor"
        if len(credit_positions) > 1:
            return None, "ambiguous_multiple_credit_rows"
        credit_detection = block[credit_positions[0]]
        after_credit = block[credit_positions[0] + 1 :]

    prerequisite_position = next(
        (
            index
            for index, item in enumerate(after_credit)
            if _is_prerequisite_detection(item)
        ),
        len(after_credit),
    )
    prerequisite_y_min = (
        after_credit[prerequisite_position].y_min
        if prerequisite_position < len(after_credit)
        else None
    )
    title_detections = [
        item
        for item in after_credit[:prerequisite_position]
        if (
            item.y_min > credit_detection.y_max
            and item.confidence >= MIN_TITLE_CONFIDENCE
            and (prerequisite_y_min is None or item.y_max < prerequisite_y_min)
            and _is_title_detection(item)
        )
    ]
    if not title_detections:
        return None, "missing_title_region"

    candidate_text = re.sub(
        r"\s+", " ", " ".join(item.text for item in title_detections)
    ).strip()
    if not candidate_text:
        return None, "empty_title_candidate"

    candidate = {
        "candidate_name_en": candidate_text,
        "candidate_confidence": min(item.confidence for item in title_detections),
        "bbox": _union_bbox(title_detections),
        "title_detections": [
            _detection_provenance(item) for item in title_detections
        ],
        "credit_detection": _detection_provenance(credit_detection),
    }
    if not _single_title_band(title_detections):
        return candidate, "unsafe_title_band"
    return candidate, None


def _source_page_details(image_path: str | Path) -> tuple[str, int | None]:
    source_page = Path(image_path).stem
    match = PAGE_RE.search(source_page)
    return source_page, int(match.group(1)) if match else None


def _provenance(
    source_page: str,
    page: int | None,
    association_status: str,
    *,
    candidate: dict[str, Any] | None = None,
    code_anchor: dict[str, Any] | None = None,
    code_occurrence: int | None = None,
    accepted: bool = False,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_page": source_page,
        "page": page,
        "status": "accepted" if accepted else "fallback",
        "association_status": association_status,
    }
    if code_anchor is not None:
        result["code_anchor"] = code_anchor
    if code_occurrence is not None:
        result["code_occurrence"] = code_occurrence
    if candidate is not None:
        result.update(
            {
                "candidate_name_en": candidate["candidate_name_en"],
                "candidate_confidence": candidate["candidate_confidence"],
                "bbox": candidate["bbox"],
                "credit_detection": candidate["credit_detection"],
                "title_detections": candidate["title_detections"],
            }
        )
    if fallback_reason is not None:
        result["fallback_reason"] = fallback_reason
    return result


def _apply_updates(
    updates: list[tuple[dict[str, Any], str | None, dict[str, Any]]],
) -> None:
    for record, candidate_name, provenance in updates:
        if candidate_name is not None:
            record["name_en"] = candidate_name
        record[PROVENANCE_KEY] = provenance


def _fallback_updates(
    courses: list[Any],
    source_page: str,
    page: int | None,
    reason: str,
) -> list[tuple[dict[str, Any], str | None, dict[str, Any]]]:
    return [
        (
            record,
            None,
            _provenance(
                source_page,
                page,
                "fallback_auxiliary_ocr_failure",
                fallback_reason=reason,
            ),
        )
        for record in courses
        if isinstance(record, dict)
    ]


def enrich_courses(
    canonical_data: dict[str, Any],
    image_path: str | Path,
    english_engine: Any,
) -> dict[str, Any]:
    """Attempt English name enrichment while preserving canonical fallbacks."""

    courses = canonical_data.get("courses", [])
    if not isinstance(courses, list):
        return canonical_data
    source_page, page = _source_page_details(image_path)

    try:
        if english_engine is None:
            _apply_updates(
                _fallback_updates(
                    courses, source_page, page, "english_engine_unavailable"
                )
            )
            return canonical_data

        ordered = ordered_detections(
            english_engine.extract_text(image_path, detail=1)
        )
        code_anchors: dict[str, list[tuple[int, Detection]]] = defaultdict(list)
        row_boundary_positions: list[int] = []
        for position, detection in enumerate(ordered):
            code = exact_numeric_code(detection.text)
            if code is None:
                if _row_boundary_like(detection.text):
                    row_boundary_positions.append(position)
                continue
            occurrence = len(code_anchors[code])
            code_anchors[code].append((position, detection))
            row_boundary_positions.append(position)

        next_row_boundary: dict[int, int | None] = {}
        for index, position in enumerate(row_boundary_positions):
            next_row_boundary[position] = (
                row_boundary_positions[index + 1]
                if index + 1 < len(row_boundary_positions)
                else None
            )

        canonical_occurrences: dict[str, int] = defaultdict(int)
        updates: list[tuple[dict[str, Any], str | None, dict[str, Any]]] = []
        for record in courses:
            if not isinstance(record, dict):
                continue
            raw_code = str(record.get("code", ""))
            code = exact_numeric_code(raw_code)
            if code is None:
                updates.append(
                    (
                        record,
                        None,
                        _provenance(
                            source_page,
                            page,
                            "fallback_non_exact_code",
                            fallback_reason="non_exact_or_composite_code",
                        ),
                    )
                )
                continue

            occurrence = canonical_occurrences[code]
            canonical_occurrences[code] += 1
            anchors = code_anchors.get(code, [])
            if occurrence >= len(anchors):
                updates.append(
                    (
                        record,
                        None,
                        _provenance(
                            source_page,
                            page,
                            "fallback_missing_exact_code_occurrence",
                            fallback_reason="missing_exact_code_occurrence",
                        ),
                    )
                )
                continue

            anchor_position, anchor = anchors[occurrence]
            candidate, skip_reason = candidate_from_anchor(
                ordered,
                anchor_position,
                next_row_boundary.get(anchor_position),
            )
            if candidate is None:
                updates.append(
                    (
                        record,
                        None,
                        _provenance(
                            source_page,
                            page,
                            "fallback_unsafe_candidate",
                            code_anchor=_detection_provenance(anchor),
                            code_occurrence=occurrence,
                            fallback_reason=skip_reason or "unsafe_candidate",
                        ),
                    )
                )
                continue

            if skip_reason is not None:
                updates.append(
                    (
                        record,
                        None,
                        _provenance(
                            source_page,
                            page,
                            "fallback_unsafe_candidate",
                            candidate=candidate,
                            code_anchor=_detection_provenance(anchor),
                            code_occurrence=occurrence,
                            fallback_reason=skip_reason,
                        ),
                    )
                )
                continue

            updates.append(
                (
                    record,
                    candidate["candidate_name_en"],
                    _provenance(
                        source_page,
                        page,
                        "associated_exact_code_occurrence",
                        candidate=candidate,
                        code_anchor=_detection_provenance(anchor),
                        code_occurrence=occurrence,
                        accepted=True,
                    ),
                )
            )

        _apply_updates(updates)
    except Exception as exc:
        _apply_updates(
            _fallback_updates(
                courses,
                source_page,
                page,
                f"{type(exc).__name__}: auxiliary OCR failed",
            )
        )

    return canonical_data
