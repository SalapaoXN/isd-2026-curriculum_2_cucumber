import argparse
import json
from pathlib import Path
from typing import Sequence

from src.rules_policy_mapper import RulesPolicyMapper


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Map extracted Academic Rules into deterministic policy categories."
    )
    parser.add_argument("input_path", type=str, help="Extracted Rules JSON path.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Optional output JSON path; otherwise print JSON to stdout.",
    )
    args = parser.parse_args(argv)

    result = RulesPolicyMapper().map_file(args.input_path)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    main()
