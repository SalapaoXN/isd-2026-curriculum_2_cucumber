"""Extraction tool: OCR result -> extracted information (incl. rule/file helpers)."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from src.pipeline.tools.extraction.engine import CurriculumExtractor, prediction_description
from src.pipeline.config import plan_label, resolve_plan, resolve_program


SOURCE_VERIFIED_CREDIT_CORRECTIONS_PATH = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "corrections"
    / "source_verified_credit_corrections.json"
)
_AUTHORITATIVE_CREDIT_RE = re.compile(r"^\d+\(\d+-\d+-\d+\)$")
_REQUIRED_CREDIT_CORRECTION_FIELDS = {
    "program",
    "plan",
    "course_code",
    "credits",
    "source_verified",
    "source_filename",
    "source_page",
    "document_category",
}


def _load_source_verified_credit_corrections(path=None):
    """Load reviewed source corrections keyed by exact source identity."""
    correction_path = (
        SOURCE_VERIFIED_CREDIT_CORRECTIONS_PATH
        if path is None
        else Path(path)
    )
    try:
        with correction_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"cannot load source-verified credit corrections: {correction_path}"
        ) from error

    records = payload.get("corrections") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("source-verified credit corrections must contain a corrections list")

    lookup = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != _REQUIRED_CREDIT_CORRECTION_FIELDS:
            raise ValueError(f"invalid source-verified credit correction at index {index}")
        if (
            not isinstance(record["program"], str)
            or not record["program"].strip()
            or not isinstance(record["plan"], str)
            or not record["plan"].strip()
            or not isinstance(record["course_code"], str)
            or not record["course_code"].strip()
            or not isinstance(record["credits"], str)
            or not _AUTHORITATIVE_CREDIT_RE.fullmatch(record["credits"].strip())
            or record["source_verified"] is not True
            or not isinstance(record["source_filename"], str)
            or not record["source_filename"].strip()
            or isinstance(record["source_page"], bool)
            or not isinstance(record["source_page"], int)
            or not isinstance(record["document_category"], str)
            or not record["document_category"].strip()
        ):
            raise ValueError(f"invalid source-verified credit correction at index {index}")

        key = (
            record["program"].strip().upper(),
            record["plan"].strip(),
            record["course_code"].strip(),
            record["source_filename"].strip(),
            record["source_page"],
            record["document_category"].strip(),
        )
        if key in lookup and lookup[key] != record:
            raise ValueError(f"ambiguous source-verified credit correction at index {index}")
        lookup[key] = dict(record)
    return lookup


def _reconcile_source_backed_credit(course, correction_lookup, *, program, plan):
    """Apply only an exact reviewed source correction; otherwise fail closed."""
    if not isinstance(course, dict):
        return course

    current_credit = course.get("credits")
    current_credit = current_credit.strip() if isinstance(current_credit, str) else ""
    course_code = course.get("code")
    source_entries = course.get("source_provenance")
    if not isinstance(course_code, str) or not isinstance(source_entries, list):
        return course

    matching_key = next(
        (
            key
            for key in correction_lookup
            if key[0] == str(program).upper()
            and key[1] == plan
            and key[2] == course_code
            and any(
                isinstance(entry, dict)
                and str(entry.get("program", "")).upper() == key[0]
                and entry.get("source_filename") == key[3]
                and entry.get("source_page") == key[4]
                and entry.get("document_category") == key[5]
                for entry in source_entries
            )
        ),
        None,
    )
    if matching_key is None:
        return course

    correction = correction_lookup[matching_key]
    corrected_credit = correction["credits"].strip()
    if current_credit and _AUTHORITATIVE_CREDIT_RE.fullmatch(current_credit):
        return course

    repaired = dict(course)
    repaired["credits"] = corrected_credit
    repaired["credit_source_verified"] = True
    repaired["credit_source_provenance"] = {
        "source_filename": correction["source_filename"],
        "source_page": correction["source_page"],
        "document_category": correction["document_category"],
    }
    repaired["source_provenance"] = [
        {
            **entry,
            "source_verified": True,
        }
        if (
            isinstance(entry, dict)
            and str(entry.get("program", "")).upper() == matching_key[0]
            and entry.get("source_filename") == matching_key[3]
            and entry.get("source_page") == matching_key[4]
            and entry.get("document_category") == matching_key[5]
        )
        else entry
        for entry in source_entries
    ]
    return repaired


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
    input_indexes = [i for i, part in enumerate(parts) if part.casefold() in {"inputs", "input"}]
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
        default="data/output/extracted",
        help="Root directory to save extracted JSON output (default: 'data/output/extracted')"
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


def run_extraction(
    input_path: str | Path,
    output_dir: str | Path,
    program: str | None = None,
    plan: str | None = None,
    prefix: str | None = None,
    pages: str | None = None,
    source: str | None = None,
) -> Path:
    """Pure extraction stage (no argv): OCR files -> per-file JSON + legacy summary."""
    input_path = Path(input_path)
    output_root = Path(output_dir)

    program_input = input_path if input_path.is_dir() else input_path.parent
    try:
        resolved_program = resolve_program(program, program_input)
        resolved_plan = resolve_plan(plan, resolved_program)
    except ValueError as exc:
        raise ValueError(f"Error: {exc}") from exc

    resolved_output_dir = output_root / resolved_program.casefold()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    extractor = CurriculumExtractor(
        program=resolved_program,
        plan=resolved_plan,
        source=source
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
        prefix,
    )

    # Filter by source page
    if pages is not None:
        try:
            selected_pages = _parse_pages(pages)
        except ValueError as exc:
            raise ValueError(f"Error: {exc}") from exc

        files_to_process = _filter_files_by_pages(
            files_to_process,
            selected_pages,
        )

    if not files_to_process:
        print(f" No valid .txt or .json files found at {input_path}")
        return resolved_output_dir

    print(f" Found {len(files_to_process)} file(s) to process.")

    all_courses = []
    last_result = None
    source_verified_corrections = _load_source_verified_credit_corrections()

    for file in files_to_process:
        # Ignore already extracted files
        if file.name.endswith("_extracted.json"):
            continue

        print(f"\nProcessing OCR output: {file.name}")
        result = extractor.process_file(file)
        result["courses"] = [
            _reconcile_source_backed_credit(
                course,
                source_verified_corrections,
                program=resolved_program,
                plan=resolved_plan,
            )
            for course in result.get("courses", [])
        ]
        last_result = result
        all_courses.extend(result["courses"])

        # Save individual extracted file
        out_file = resolved_output_dir / f"{file.stem}_extracted.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f" Saved extracted JSON: {out_file}")

    # If processing multiple files, also output a merged summary file
    if len(files_to_process) > 1 and last_result:
        merged_result = {
            "source": last_result.get("source", source),
            "description": prediction_description(resolved_program, resolved_plan, consolidated=True),
            "program": resolved_program,
            "plan": resolved_plan,
            "courses": all_courses
        }
        group_id = _input_group_identifier(input_path)
        program_id = _safe_identifier(resolved_program)
        plan_id = _safe_identifier(plan_label(resolved_plan))
        legacy_output_dir = resolved_output_dir / "legacy" / "consolidated_summaries"
        legacy_output_dir.mkdir(parents=True, exist_ok=True)
        consolidated_file = legacy_output_dir / f"consolidated_curriculum_{group_id}_{program_id}_{plan_id}.json"
        with open(consolidated_file, "w", encoding="utf-8") as f:
            json.dump(merged_result, f, ensure_ascii=False, indent=4)
        print(f"\n Saved consolidated result ({len(all_courses)} courses total): {consolidated_file}")

    return resolved_output_dir


def main():
    args = parse_arguments()
    try:
        run_extraction(
            args.input_path,
            args.output_dir,
            program=args.program,
            plan=args.plan,
            prefix=args.prefix,
            pages=args.pages,
            source=args.source,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
