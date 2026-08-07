import argparse
import json
from pathlib import Path
from src import CurriculumExtractor


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract structured course and curriculum data from OCR outputs."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to OCR .txt/.json file or folder containing OCR files."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save extracted JSON output (default: 'outputs')"
    )
    parser.add_argument(
        "--program",
        type=str,
        default="DSBA",
        help="Program name (default: 'DSBA')"
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="coop",
        help="Study plan name (e.g. 'coop', 'regular') (default: 'coop')"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="GT_Template-2.xlsx / Academic Plan GT — DSBA coop",
        help="Source label for metadata"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = CurriculumExtractor(
        program=args.program,
        plan=args.plan,
        source=args.source
    )

    files_to_process = []
    if input_path.is_dir():
        files_to_process = list(input_path.glob("*.txt")) + list(input_path.glob("*.json"))
    else:
        files_to_process = [input_path]

    if not files_to_process:
        print(f"❌ No valid .txt or .json files found at {input_path}")
        return

    print(f"🔍 Found {len(files_to_process)} file(s) to process.")

    all_courses = []
    last_result = None

    for file in files_to_process:
        # Ignore already extracted files
        if file.name.endswith("_extracted.json"):
            continue

        print(f"\nProcessing OCR output: {file.name}")
        result = extractor.process_file(file)
        last_result = result
        all_courses.extend(result["courses"])

        # Save individual extracted file
        out_file = output_dir / f"{file.stem}_extracted.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"✓ Saved extracted JSON: {out_file}")

    # If processing multiple files, also output a merged summary file
    if len(files_to_process) > 1 and last_result:
        merged_result = {
            "source": args.source,
            "description": f"Ground Truth รายวิชาหลักสูตร {args.program} (แผน {args.plan}) - Consolidated",
            "program": args.program,
            "plan": args.plan,
            "courses": all_courses
        }
        consolidated_file = output_dir / f"consolidated_curriculum_{args.program}_{args.plan}.json"
        with open(consolidated_file, "w", encoding="utf-8") as f:
            json.dump(merged_result, f, ensure_ascii=False, indent=4)
        print(f"\n✓ Saved consolidated result ({len(all_courses)} courses total): {consolidated_file}")


if __name__ == "__main__":
    main()