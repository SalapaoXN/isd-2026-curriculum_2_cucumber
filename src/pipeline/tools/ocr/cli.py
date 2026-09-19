"""User-facing wrapper for the standalone OCR stage (clean-layout port of ocr.py)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.config import normalize_program
from src.pipeline.tools.ocr.pipeline_runner import parse_pages, run_ocr


BASE_DIR = Path(__file__).resolve().parents[4]
SUPPORTED_PREFIXES = ("ait", "bit", "dsba", "gened", "it")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone curriculum OCR.")
    parser.add_argument(
        "--prefix",
        required=True,
        help="Input corpus prefix: ait, bit, dsba, gened, or it.",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Optional pages or ranges, for example 32-38 or 16,17,20.",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    prefix = str(args.prefix).strip().casefold()
    if prefix not in SUPPORTED_PREFIXES:
        supported = ", ".join(SUPPORTED_PREFIXES)
        raise SystemExit(f"Error: Unsupported prefix '{args.prefix}'. Choose one of: {supported}.")

    input_dir = BASE_DIR / "data" / "input" / prefix
    output_dir = BASE_DIR / "data" / "output"
    try:
        pages = parse_pages(args.pages) if args.pages is not None else None
        run_ocr(
            input_dir=input_dir,
            output_dir=output_dir,
            program=normalize_program(prefix),
            pages=pages,
            no_gpu=args.no_gpu,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
