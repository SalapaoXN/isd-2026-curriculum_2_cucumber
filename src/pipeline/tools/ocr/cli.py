"""Dataset-aware standalone OCR wrapper.

Normal end-to-end project usage should prefer ``python -m src.pipeline.run``.
This module remains useful when only the OCR artifacts are needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.datasets import DATASET_CONFIG, get_dataset_config
from src.pipeline.tools.ocr.pipeline_runner import parse_pages, run_ocr


BASE_DIR = Path(__file__).resolve().parents[4]
SUPPORTED_PREFIXES = tuple(DATASET_CONFIG)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone curriculum OCR.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        choices=SUPPORTED_PREFIXES,
        help="Curriculum dataset key, for example it2560 or it2565.",
    )
    # Old spelling kept as a quiet compatibility alias.
    group.add_argument("--prefix", help=argparse.SUPPRESS)
    parser.add_argument("--pages", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    requested = args.dataset or args.prefix
    try:
        config = get_dataset_config(requested)
        input_dir = config.resolve_input_dir(BASE_DIR)
        pages = (
            parse_pages(args.pages)
            if args.pages is not None
            else config.required_pages()
        )
        run_ocr(
            input_dir=input_dir,
            output_dir=BASE_DIR / "data" / "output",
            program=config.program,
            pages=pages,
            no_gpu=args.no_gpu,
            dataset_key=config.key,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
