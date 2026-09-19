import argparse
from pathlib import Path
from typing import List

from src.pipeline.utils.file_handler import save_ocr_results
from src.pipeline.tools.ocr.engine import OCREngine
from src.pipeline.utils.page_metadata import resolve_document_page
from src.pipeline.utils.pre_clean import pre_clean_with_regex
from src.pipeline.config import (
    discover_page_files,
    discover_pages,
    plan_label,
    resolve_plan,
    resolve_program,
)

BASE_DIR = Path(__file__).resolve().parents[4]


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
    """Convert a page spec such as ``32-36`` or ``16,17,20`` to page numbers."""
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
        description="Run the standalone OCR stage for specific pages."
    )
    parser.add_argument(
        "-p", "--pages",
        type=str,
        default=None,
        help=(
            "Specify pages or ranges (e.g. '32-36', '16,17,20', '16'). "
            "If omitted, discover direct-child page images automatically."
        )
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default=str(BASE_DIR / "data" / "input" / "dsba"),
        help="Directory containing images (default: 'data/input/dsba')"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=str(BASE_DIR / "data" / "output"),
        help="Root directory to save OCR and extracted outputs (default: 'data/output')"
    )
    parser.add_argument(
        "--program",
        type=str,
        default=None,
        help="Program name; derived from a supported input directory when omitted"
    )
    parser.add_argument(
        "--plan",
        type=str,
        default=None,
        help="Study plan: coop, no_coop, or gened; required where applicable"
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
) -> Path:
    """Run only OCR and pre-cleaning, returning the OCR output directory."""
    if pages is None:
        pages = discover_pages(input_dir)
    if not pages:
        raise ValueError(
            "No valid pages were requested. Use a page number or range such as -p 32-36."
        )

    page_files = discover_page_files(input_dir)
    if not any(page in page_files for page in pages):
        raise ValueError(f"None of the requested pages were found in '{input_dir}'.")

    ocr_output_dir = output_dir / "ocr" / program.casefold()
    ocr_output_dir.mkdir(parents=True, exist_ok=True)

    print(" Starting the standalone OCR stage")
    print(f" Pages to process: {pages}")
    print(f" Program: {program}; plan: {plan_label(plan)}")

    use_gpu = not no_gpu
    engine = OCREngine(languages=["th", "en"], gpu=use_gpu)

    for page_num in pages:
        base_name = f"{input_dir.name}_page_{page_num:03d}"

        img_file = page_files.get(page_num)

        if not img_file:
            print(f"\n  [Skip] No image file found for page {page_num} in '{input_dir}'")
            continue

        print(f"\n========================================")
        print(f" Processing: {img_file.name}")
        print(f"========================================")

        # Step 1: OCR
        lines = engine.extract_text(img_file, detail=0)
        print(f"   ├─ OCR read {len(lines)} lines")

        lines = [line.upper() for line in lines]

        # Step 1.5: Deterministic regex pre-clean (before extraction)
        raw_text_joined = "\n".join(lines)
        raw_text_joined = pre_clean_with_regex(raw_text_joined)
        lines = [line for line in raw_text_joined.split("\n") if line.strip()]
        document_page = _document_page_for_ocr(
            lines, program, page_num, source_filename=img_file.name
        )

        # Step 1.5: Autocorrect English text (pyspellchecker)
        # lines, typos = spell_checker.process_lines(lines)
        # print(f"   ├─ Autocorrect fixed {len(typos)} points")
        # if typos:
        #     for t in typos:
        #         print(f"   │    L{t['line']}: {t['original']} -> {t['corrected']}")

        save_ocr_results(
            lines,
            ocr_output_dir,
            base_name,
            source_filename=img_file.name,
            source_page=page_num,
            program=program,
            document_page=document_page,
        )

    print(f"\n Finished OCR stage! Files saved at: {ocr_output_dir.resolve()}")
    return ocr_output_dir


def main():
    args = parse_arguments()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    try:
        program = resolve_program(args.program, input_dir)
        plan = resolve_plan(args.plan, program)
        pages = parse_pages(args.pages) if args.pages is not None else discover_pages(input_dir)
        run_ocr(
            input_dir=input_dir,
            output_dir=output_dir,
            program=program,
            pages=pages,
            no_gpu=args.no_gpu,
            plan=plan,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
