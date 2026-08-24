"""Non-production DSBA English-only OCR second-pass prototype.

The canonical Thai+English OCR and extractor remain authoritative. This module
only emits English candidates with enough provenance to review their association
with canonical records.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "inputs" / "dsba"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "prototype_outputs" / "dsba_english_second_pass"

CODE_RE = re.compile(r"^\d{8}$")
PLACEHOLDER_CODE_RE = re.compile(r"^\d{4,8}X+$")
CREDIT_RE = re.compile(r"\d+\s*\(\s*\d+\s*[-–]\s*\d+\s*[-–]\s*\d+\s*\)")
PREREQUISITE_RE = re.compile(r"^PRERE[A-Z]*\b", re.IGNORECASE)
HAS_ASCII_LETTER_RE = re.compile(r"[A-Z]", re.IGNORECASE)
MIN_TITLE_CONFIDENCE = 0.5

_TORCH_DLL_DIRECTORY_HANDLE: Any = None


@dataclass(frozen=True)
class Detection:
    """One EasyOCR detail=1 detection in spatial reading order."""

    original_index: int
    bbox: list[list[float]]
    text: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def _load_torch() -> Any:
    """Load Torch after registering its Windows DLL directory if necessary."""

    global _TORCH_DLL_DIRECTORY_HANDLE

    if os.name == "nt":
        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            torch_lib = Path(spec.origin).parent / "lib"
            if torch_lib.is_dir():
                _TORCH_DLL_DIRECTORY_HANDLE = os.add_dll_directory(str(torch_lib))

    import torch

    return torch


def require_gpu() -> Any:
    """Fail before OCR if CUDA is unavailable; never silently use CPU."""

    torch = _load_torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU OCR is required for this prototype, but torch.cuda.is_available() "
            "is false. No CPU fallback is permitted."
        )
    if torch.cuda.device_count() < 1:
        raise RuntimeError("GPU OCR is required, but no CUDA device is visible.")
    return torch


def parse_pages(value: str) -> list[int]:
    """Parse explicit page numbers/ranges without discovering other inputs."""

    pages: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid page range: {token}")
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(token))

    if not pages:
        raise ValueError("At least one explicit DSBA page is required.")
    return list(dict.fromkeys(pages))


def exact_numeric_code(value: Any) -> str | None:
    """Return only an exact normalized eight-digit code; never repair OCR."""

    text = re.sub(r"\s+", "", str(value or "").strip())
    return text if CODE_RE.fullmatch(text) else None


def placeholder_code_like(value: Any) -> bool:
    """Return whether a normalized placeholder code can delimit a row."""

    text = re.sub(r"\s+", "", str(value or "").strip()).upper()
    return bool(PLACEHOLDER_CODE_RE.fullmatch(text))


def _bbox_values(bbox: Iterable[Iterable[Any]]) -> tuple[list[list[float]], float, float, float, float]:
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

    return sorted(detections, key=lambda item: (item.y_min, item.x_min, item.original_index))


def _is_code_detection(detection: Detection) -> str | None:
    return exact_numeric_code(detection.text)


def _is_credit_detection(detection: Detection) -> bool:
    return bool(CREDIT_RE.search(detection.text))


def _is_prerequisite_detection(detection: Detection) -> bool:
    return bool(PREREQUISITE_RE.match(detection.text.strip()))


def _is_title_detection(detection: Detection) -> bool:
    text = detection.text.strip()
    if (
        not text
        or exact_numeric_code(text)
        or placeholder_code_like(text)
        or _is_credit_detection(detection)
    ):
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


def candidate_from_anchor(
    ordered: list[Detection],
    anchor_position: int,
    next_code_position: int | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Extract one title region after a code/credit pair.

    The region ends at the next code anchor or prerequisite heading. Multiple
    credit rows are treated as ambiguous because they can represent alternatives.
    """

    end = next_code_position if next_code_position is not None else len(ordered)
    block = ordered[anchor_position + 1 : end]
    preceding_credit = (
        ordered[anchor_position - 1]
        if anchor_position > 0 and _is_credit_detection(ordered[anchor_position - 1])
        else None
    )
    if preceding_credit is not None:
        # Detail=1 table detections can sort the right-column credit immediately
        # before the left-column code by y/x position.
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
            and (
                prerequisite_y_min is None or item.y_max < prerequisite_y_min
            )
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

    return (
        {
            "english_candidate_name_en": candidate_text,
            "candidate_confidence": min(item.confidence for item in title_detections),
            "bbox": _union_bbox(title_detections),
            "title_detections": [
                _detection_provenance(item) for item in title_detections
            ],
            "credit_detection": _detection_provenance(credit_detection),
        },
        None,
    )


def _canonical_lines_for_extractor(lines: list[str]) -> list[str]:
    from src.pre_clean import pre_clean_with_regex

    upper_lines = [str(line).upper() for line in lines]
    cleaned = pre_clean_with_regex("\n".join(upper_lines))
    return [line for line in cleaned.split("\n") if line.strip()]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _page_image(input_dir: Path, page: int) -> Path:
    if input_dir.name.lower() != "dsba":
        raise ValueError("This prototype accepts only an input directory named 'dsba'.")
    image = input_dir / f"dsba_page_{page:03d}.png"
    if not image.is_file():
        raise FileNotFoundError(f"Targeted DSBA image not found: {image}")
    return image


def _validate_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    prototype_root = DEFAULT_OUTPUT_DIR.resolve()
    if not resolved.is_relative_to(prototype_root):
        raise ValueError(
            "Prototype output must remain under prototype_outputs/dsba_english_second_pass."
        )
    return resolved


def process_page(
    page: int,
    image_path: Path,
    output_dir: Path,
    canonical_engine: Any,
    english_engine: Any,
    plan: str,
) -> dict[str, Any]:
    from src.extractor import CurriculumExtractor

    base_name = image_path.stem
    raw_canonical_lines = canonical_engine.extract_text(image_path, detail=0)
    canonical_lines = _canonical_lines_for_extractor(raw_canonical_lines)
    canonical_ocr_path = output_dir / f"{base_name}_canonical_ocr.json"
    _write_json(
        canonical_ocr_path,
        {
            "filename": base_name,
            "line_count": len(canonical_lines),
            "text_lines": canonical_lines,
        },
    )

    extractor = CurriculumExtractor(program="DSBA", plan=plan)
    canonical_data = extractor.process_file(canonical_ocr_path)

    english_results = english_engine.extract_text(image_path, detail=1)
    ordered = ordered_detections(english_results)
    code_anchors: dict[str, list[tuple[int, Detection]]] = defaultdict(list)
    row_boundary_positions: list[int] = []
    for position, detection in enumerate(ordered):
        code = _is_code_detection(detection)
        if code is None:
            if placeholder_code_like(detection.text):
                row_boundary_positions.append(position)
            continue
        occurrence = len(code_anchors[code])
        code_anchors[code].append((position, detection))
        row_boundary_positions.append(position)

    next_code_position: dict[int, int | None] = {}
    for index, position in enumerate(row_boundary_positions):
        next_code_position[position] = (
            row_boundary_positions[index + 1]
            if index + 1 < len(row_boundary_positions)
            else None
        )

    canonical_occurrences: dict[str, int] = defaultdict(int)
    associations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for record_index, record in enumerate(canonical_data.get("courses", [])):
        raw_code = str(record.get("code", ""))
        code = exact_numeric_code(raw_code)
        if code is None:
            skipped.append(
                {
                    "record_index": record_index,
                    "code": raw_code,
                    "association_status": "skipped_placeholder_or_alternative_code",
                }
            )
            continue

        occurrence = canonical_occurrences[code]
        canonical_occurrences[code] += 1
        anchors = code_anchors.get(code, [])
        if occurrence >= len(anchors):
            skipped.append(
                {
                    "record_index": record_index,
                    "code": code,
                    "occurrence": occurrence,
                    "association_status": "skipped_missing_exact_code_occurrence",
                }
            )
            continue

        anchor_position, anchor = anchors[occurrence]
        candidate, skip_reason = candidate_from_anchor(
            ordered,
            anchor_position,
            next_code_position.get(anchor_position),
        )
        if candidate is None:
            skipped.append(
                {
                    "record_index": record_index,
                    "code": code,
                    "occurrence": occurrence,
                    "association_status": f"skipped_{skip_reason}",
                    "code_anchor": _detection_provenance(anchor),
                }
            )
            continue

        associations.append(
            {
                "record_index": record_index,
                "code": code,
                "occurrence": occurrence,
                "canonical_course": record,
                "canonical_name_en": record.get("name_en", ""),
                "english_candidate_name_en": candidate["english_candidate_name_en"],
                "candidate_confidence": candidate["candidate_confidence"],
                "bbox": candidate["bbox"],
                "association_status": "associated_exact_code_occurrence",
                "provenance": {
                    "source_page": base_name,
                    "association_basis": "normalized exact 8-digit code plus occurrence order",
                    "code_anchor": _detection_provenance(anchor),
                    "code_occurrence": occurrence,
                    "credit_detection": candidate["credit_detection"],
                    "title_detections": candidate["title_detections"],
                },
            }
        )

    page_result = {
        "source_page": base_name,
        "program": "DSBA",
        "plan": canonical_data.get("plan"),
        "ocr_settings": {
            "canonical_languages": ["th", "en"],
            "canonical_gpu": True,
            "canonical_detail": 0,
            "english_languages": ["en"],
            "english_gpu": True,
            "english_detail": 1,
            "preprocessing_changes": False,
        },
        "canonical": canonical_data,
        "associations": associations,
        "skipped": skipped,
        "english_detection_count": len(ordered),
    }
    _write_json(output_dir / f"{base_name}_second_pass.json", page_result)
    return page_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the non-production DSBA English-only OCR second-pass prototype."
    )
    parser.add_argument("--pages", required=True, help="Explicit pages, e.g. 026,027,039")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plan", choices=("coop", "no_coop"), default="coop")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pages = parse_pages(args.pages)
    input_dir = args.input_dir
    output_dir = _validate_output_dir(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"DSBA input directory not found: {input_dir}")

    require_gpu()

    from src.ocr_engine import OCREngine

    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_engine = OCREngine(languages=["th", "en"], gpu=True)
    english_engine = OCREngine(languages=["en"], gpu=True)

    page_results = []
    for page in pages:
        image_path = _page_image(input_dir, page)
        page_results.append(
            process_page(
                page,
                image_path,
                output_dir,
                canonical_engine,
                english_engine,
                args.plan,
            )
        )

    _write_json(
        output_dir / "summary.json",
        {
            "program": "DSBA",
            "pages": pages,
            "output_dir": str(output_dir),
            "canonical_fields_mutated": False,
            "candidate_replacement_performed": False,
            "associated_count": sum(len(item["associations"]) for item in page_results),
            "skipped_count": sum(len(item["skipped"]) for item in page_results),
        },
    )
    return 0


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
