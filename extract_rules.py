import argparse
import json
from pathlib import Path
from typing import Sequence

from src.rule_extractor import RuleExtractor, discover_rule_ocr_files


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract numbered academic rules from OCR TXT/JSON files."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Rule OCR TXT/JSON file or directory containing Rule OCR files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output-file",
        default="rules_extracted.json",
        help="Consolidated Rules JSON output path (default: rules_extracted.json).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = parse_arguments()
    args = parser.parse_args(argv)

    try:
        files = discover_rule_ocr_files(args.input_path)
        result = RuleExtractor().extract_from_files(files)
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    print(
        f"Extracted {result['total_rules']} rule(s) from {len(files)} OCR file(s) "
        f"to {output_path}"
    )
    return 0


if __name__ == "__main__":
    main()
