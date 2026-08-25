import argparse
import hashlib
import json
import re
from pathlib import Path
from src.extractor import CurriculumExtractor, prediction_description
from src.pipeline_config import plan_label, resolve_plan, resolve_program


def _safe_identifier(value: str, fallback: str = "input") -> str:
    raw = "" if value is None else str(value)
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    if not raw:
        return fallback
    if not cleaned:
        cleaned = fallback
    if cleaned != raw:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        cleaned = f"{cleaned}_{digest}"
    return cleaned


def _input_group_identifier(input_path: Path) -> str:
    parts = list(input_path.parts)
    input_indexes = [i for i, part in enumerate(parts) if part.casefold() == "inputs"]
    if input_indexes:
        parts = parts[input_indexes[-1] + 1 :]
        source = "/".join(parts) or input_path.name
        return _safe_identifier(source)

    source = "/".join(parts[-2:]) or input_path.name
    readable = _safe_identifier(source)
    normalized_path = str(input_path.expanduser().resolve())
    path_digest = hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:12]
    return f"{readable}_{path_digest}"

def _filter_files_by_prefix(files, prefix):
    if not prefix:
        return list(files)

    normalized_prefix = prefix.casefold()
    return [
        file
        for file in files
        if file.name.casefold().startswith(normalized_prefix)
    ]

def _parse_pages(page_input):
    pages = set()

    for part in page_input.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            bounds = part.split("-")
            if (
                len(bounds) != 2
                or not bounds[0].strip().isdigit()
                or not bounds[1].strip().isdigit()
            ):
                raise ValueError(
                    f"Invalid page range '{part}'. Use a range such as 26-32."
                )

            start = int(bounds[0].strip())
            end = int(bounds[1].strip())

            if start > end:
                raise ValueError(
                    f"Invalid descending page range '{part}'."
                )

            pages.update(range(start, end + 1))

        elif part.isdigit():
            pages.add(int(part))

        else:
            raise ValueError(
                f"Invalid page value '{part}'."
            )

    return pages


def _filter_files_by_pages(files, pages):
    if pages is None:
        return list(files)

    result = []

    for file in files:
        match = re.search(
            r"page_(\d+)",
            file.name,
            re.IGNORECASE,
        )

        if match and int(match.group(1)) in pages:
            result.append(file)

    return result

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
        "--prefix",
        type=str,
        default=None,
        help="Only process OCR files whose filenames start with this prefix"
    )
    parser.add_argument(
        "-p",
        "--pages",
        type=str,
        default=None,
        help="Only process selected OCR pages, e.g. 26-32 or 26-32,44-117"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source label for metadata"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    program_input = input_path if input_path.is_dir() else input_path.parent
    try:
        program = resolve_program(args.program, program_input)
        plan = resolve_plan(args.plan, program)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    extractor = CurriculumExtractor(
        program=program,
        plan=plan,
        source=args.source
    )

    files_to_process = []
    if input_path.is_dir():
        files_by_stem = {
            file.stem: file
            for file in input_path.glob("*.txt")
        }
        for file in input_path.glob("*.json"):
            if not file.name.endswith("_extracted.json"):
                files_by_stem[file.stem] = file
        files_to_process = list(files_by_stem.values())
    else:
        files_to_process = [input_path]
    
    # Filter by filename prefix
    files_to_process = _filter_files_by_prefix(
        files_to_process,
        args.prefix,
    )

    # Filter by source page
    if args.pages is not None:
        try:
            selected_pages = _parse_pages(args.pages)
        except ValueError as exc:
            raise SystemExit(f"Error: {exc}") from exc

        files_to_process = _filter_files_by_pages(
            files_to_process,
            selected_pages,
        )

    if not files_to_process:
        print(f" No valid .txt or .json files found at {input_path}")
        return

    print(f" Found {len(files_to_process)} file(s) to process.")

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
        print(f" Saved extracted JSON: {out_file}")

    # If processing multiple files, also output a merged summary file
    if len(files_to_process) > 1 and last_result:
        merged_result = {
            "source": last_result.get("source", args.source),
            "description": prediction_description(program, plan, consolidated=True),
            "program": program,
            "plan": plan,
            "courses": all_courses
        }
        group_id = _input_group_identifier(input_path)
        program_id = _safe_identifier(program)
        plan_id = _safe_identifier(plan_label(plan))
        consolidated_file = output_dir / f"consolidated_curriculum_{group_id}_{program_id}_{plan_id}.json"
        with open(consolidated_file, "w", encoding="utf-8") as f:
            json.dump(merged_result, f, ensure_ascii=False, indent=4)
        print(f"\n Saved consolidated result ({len(all_courses)} courses total): {consolidated_file}")


if __name__ == "__main__":
    main()
