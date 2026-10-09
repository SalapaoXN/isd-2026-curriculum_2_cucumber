import argparse
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, List

from PIL import Image, UnidentifiedImageError

from src.pipeline.utils.file_handler import save_ocr_results
from src.pipeline.tools.ocr.engine import OCREngine
from src.pipeline.utils.page_metadata import resolve_document_page
from src.pipeline.utils.pre_clean import pre_clean_with_regex
from src.pipeline.config import (
    discover_page_files,
    discover_pages,
    resolve_program,
)

BASE_DIR = Path(__file__).resolve().parents[4]


def _validate_dataset_key(dataset_key: str | None) -> str | None:
    """Require an explicit dataset key to be one safe directory component."""
    if dataset_key is None:
        return None
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
    if (
        not isinstance(dataset_key, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", dataset_key) is None
        or dataset_key.upper() in reserved_names
    ):
        raise ValueError(
            f"Invalid dataset key {dataset_key!r}; use a non-empty, path-safe "
            "single directory name containing only letters, digits, '_' or '-'."
        )
    return dataset_key


def _normalize_ocr_detections(detections: Any) -> list[dict[str, Any]]:
    """Normalize EasyOCR records while retaining string-only test adapters."""
    if not isinstance(detections, list):
        raise ValueError("OCR engine returned an invalid detection collection")

    normalized: list[dict[str, Any]] = []
    for detection in detections:
        if isinstance(detection, str):
            normalized.append({"text": detection})
            continue
        if isinstance(detection, Mapping) and isinstance(detection.get("text"), str):
            normalized.append(dict(detection))
            continue
        raise ValueError("OCR engine returned an invalid detection record")
    return normalized


def _document_page_for_ocr(
    text_lines: List[str],
    program: str,
    source_page: int,
    source_filename: str | None = None,
) -> int | None:
    """Compatibility wrapper around the central document-page resolver."""
    return resolve_document_page(
        program,
        source_page,
        source_filename,
        text_lines,
    ).document_page


def parse_pages(pages_str: str) -> List[int]:
    """Convert an internal page spec such as ``32-36`` to page numbers."""
    pages = set()
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            continue

        separator = ".." if ".." in part else ("-" if "-" in part else None)
        if separator is not None:
            bounds = part.split(separator)
            if len(bounds) != 2 or not all(bound.strip().isdigit() for bound in bounds):
                raise ValueError(
                    f"Invalid page range '{part}'. Use a range such as 32-36."
                )
            start, end = (int(bound.strip()) for bound in bounds)
            if start > end:
                raise ValueError(
                    f"Invalid descending page range '{part}'. Start must not exceed end."
                )
            pages.update(range(start, end + 1))
        elif part.isdigit():
            pages.add(int(part))
        else:
            raise ValueError(
                f"Invalid page value '{part}'. Use a number, range, or comma-separated list."
            )
    return sorted(list(pages))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Low-level standalone OCR runner. Normal project usage should use src.pipeline.run."
    )
    parser.add_argument(
        "-p", "--pages",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default=str(BASE_DIR / "data" / "input" / "dsba"),
        help="Directory containing source images"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=str(BASE_DIR / "data" / "output"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--program",
        type=str,
        default=None,
        help="Logical program name (IT, BIT, DSBA, AIT, or GENED)"
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default=None,
        help="Edition-specific output key such as it2560 or it2565",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU mode"
    )
    return parser.parse_args()


def run_ocr(
    input_dir: Path,
    output_dir: Path,
    program: str,
    pages: List[int] | None = None,
    no_gpu: bool = False,
    plan: str | None = None,
    dataset_key: str | None = None,
    source_prefix: str | None = None,
) -> Path:
    """Run OCR and deterministic pre-cleaning for one dataset.

    ``plan`` remains only as a compatibility parameter for older internal
    callers. OCR itself is plan-agnostic; plan routing happens in preparation.
    """
    dataset_key = _validate_dataset_key(dataset_key)
    if pages is None:
        pages = discover_pages(input_dir, prefix=source_prefix)
    if not pages:
        raise ValueError("No valid OCR pages were configured.")

    page_files = discover_page_files(input_dir, prefix=source_prefix)
    if not any(page in page_files for page in pages):
        raise ValueError(f"None of the configured pages were found in '{input_dir}'.")

    output_key = dataset_key if dataset_key is not None else program.casefold()
    ocr_output_dir = output_dir / "ocr" / output_key
    ocr_output_dir.mkdir(parents=True, exist_ok=True)

    print(" Starting OCR stage")
    print(f" Dataset: {output_key}")
    print(f" Program: {program}")
    print(f" Pages to process: {pages}")

    use_gpu = not no_gpu
    engine = OCREngine(languages=["th", "en"], gpu=use_gpu)

    for page_num in pages:
        # Generated artifact names use the explicit edition key. The actual
        # source filename remains preserved separately in provenance metadata.
        base_name = f"{output_key}_page_{page_num:03d}"
        img_file = page_files.get(page_num)

        if not img_file:
            print(f"\n  [Skip] No image file found for page {page_num} in '{input_dir}'")
            continue

        print("\n========================================")
        print(f" Processing: {img_file.name}")
        print("========================================")

        detections = _normalize_ocr_detections(
            engine.extract_text(img_file, detail=1)
        )
        lines = [detection["text"].upper() for detection in detections]
        print(f"   ├─ OCR read {len(lines)} lines")

        try:
            with Image.open(img_file) as image:
                image_width, image_height = image.size
        except UnidentifiedImageError:
            image_width = image_height = None

        raw_text_joined = "\n".join(lines)
        raw_text_joined = pre_clean_with_regex(raw_text_joined)
        lines = [line for line in raw_text_joined.split("\n") if line.strip()]
        document_page = _document_page_for_ocr(
            lines, program, page_num, source_filename=img_file.name
        )

        save_ocr_results(
            lines,
            ocr_output_dir,
            base_name,
            source_filename=img_file.name,
            source_page=page_num,
            program=program,
            document_page=document_page,
            detections=detections,
            image_width=image_width,
            image_height=image_height,
            source_dataset=output_key,
        )

    print(f"\n Finished OCR stage! Files saved at: {ocr_output_dir.resolve()}")
    return ocr_output_dir


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    try:
        program = resolve_program(args.program, input_dir)
        pages = parse_pages(args.pages) if args.pages is not None else discover_pages(input_dir)
        run_ocr(
            input_dir=input_dir,
            output_dir=output_dir,
            program=program,
            pages=pages,
            no_gpu=args.no_gpu,
            dataset_key=getattr(args, "dataset_key", None),
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
