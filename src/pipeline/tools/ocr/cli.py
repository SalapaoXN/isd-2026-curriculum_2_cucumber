"""Dataset-aware standalone OCR wrapper.

Normal end-to-end project usage should prefer ``python -m src.pipeline.run``.
The old ``--prefix`` interface remains as a compatibility/debug path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.datasets import DATASET_ALIASES, DATASET_CONFIG, get_dataset_config
from src.pipeline.tools.ocr.pipeline_runner import parse_pages, run_ocr


BASE_DIR = Path(__file__).resolve().parents[4]
SUPPORTED_DATASETS = tuple(DATASET_CONFIG)
# Compatibility constant used by older tests/callers.
SUPPORTED_PREFIXES = tuple(DATASET_ALIASES)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone curriculum OCR.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        help="Curriculum dataset key, for example it2560 or it2565.",
    )
    group.add_argument(
        "--prefix",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--pages", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU mode.")
    return parser.parse_args()


def _run_legacy_prefix(args: argparse.Namespace, prefix: str) -> None:
    # Old unversioned prefixes resolve through the dataset registry so they
    # continue to work after source folders are renamed to explicit editions.
    if prefix in DATASET_ALIASES or prefix in DATASET_CONFIG:
        config = get_dataset_config(prefix)
        run_ocr(
            input_dir=config.resolve_input_dir(BASE_DIR),
            output_dir=BASE_DIR / "data" / "output",
            program=config.program,
            pages=(
                parse_pages(args.pages)
                if args.pages is not None
                else config.required_pages()
            ),
            no_gpu=args.no_gpu,
            dataset_key=config.key,
        )
        return

    supported = ", ".join((*SUPPORTED_PREFIXES, *SUPPORTED_DATASETS))
    raise SystemExit(
        f"Error: Unsupported prefix '{args.prefix}'. Choose one of: {supported}."
    )


def main() -> None:
    args = parse_arguments()
    try:
        if args.prefix is not None:
            _run_legacy_prefix(args, str(args.prefix).strip().casefold())
            return

        config = get_dataset_config(args.dataset)
        run_ocr(
            input_dir=config.resolve_input_dir(BASE_DIR),
            output_dir=BASE_DIR / "data" / "output",
            program=config.program,
            pages=(
                parse_pages(args.pages)
                if args.pages is not None
                else config.required_pages()
            ),
            no_gpu=args.no_gpu,
            dataset_key=config.key,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
