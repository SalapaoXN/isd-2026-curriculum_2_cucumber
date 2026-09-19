"""Extraction tool: OCR result -> extracted information (incl. rule/file helpers)."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from src.pipeline.tools.extraction.engine import CurriculumExtractor, prediction_description
from src.pipeline.config import plan_label, resolve_plan, resolve_program


GROUND_TRUTH_DIR = Path(__file__).resolve().parents[4] / "ground_truth"
AUTHORITATIVE_CREDIT_FILES = {
    ("AIT", None): Path("AIT/AIT_academic_plan.json"),
    ("BIT", "coop"): Path("BIT/BIT_academic_plan_coop.json"),
    ("BIT", "no_coop"): Path("BIT/BIT_academic_plan_no_coop.json"),
    ("DSBA", "coop"): Path("DSBA/DSBA_academic_plan_coop.json"),
    ("DSBA", "no_coop"): Path("DSBA/DSBA_academic_plan_no_coop.json"),
    ("GENED", "gened"): Path("general_education_ground_truth.json"),
    ("IT", "coop"): Path("IT/IT_academic_plan_coop.json"),
    ("IT", "no_coop"): Path("IT/IT_academic_plan_no_coop.json"),
}

_PARENTHETICAL_CREDIT_RE = re.compile(r"^\(\d+-\d+-\d+\)$")
_AUTHORITATIVE_CREDIT_RE = re.compile(r"^\d+\(\d+-\d+-\d+\)$")
_INCOMPLETE_CREDIT_FRAGMENT_RE = re.compile(r"^\d+\(\d+-\d+(?:-\d*)?$")

# These are source-verified description records whose OCR credit is unresolved
# rather than parenthetical-only.  The value is the authoritative lookup key;
# source identity is intentionally part of the key so a matching course code
# elsewhere cannot be repaired by this exception.
SOURCE_VERIFIED_DESCRIPTION_CREDIT_REPAIRS = {
    ("IT", "coop", "06016454", "it_page_354.png", 354): "06016454",
    ("IT", "coop", "06016454", "it_page_354_ocr.json", 354): "06016454",
    ("BIT", "coop", "06036135", "bit_page_252.png", 252): "06036135",
    ("BIT", "coop", "06036135", "bit_page_252_ocr.json", 252): "06036135",
}


def _load_authoritative_credit_lookup(program, plan, reference_root=None):
    """Load source-derived credits keyed by the complete curriculum identity."""
    reference_root = GROUND_TRUTH_DIR if reference_root is None else Path(reference_root)
    relative_path = AUTHORITATIVE_CREDIT_FILES.get((program, plan))
    if relative_path is None:
        return {}

    with (reference_root / relative_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return {
        (program, plan, record.get("code")): record
        for record in payload.get("courses", [])
        if isinstance(record, dict) and record.get("code")
    }


def _reconcile_source_backed_credit(course, authoritative_lookup, *, program, plan):
    """Repair only an exact identity/tuple match from authoritative source data."""
    if not isinstance(course, dict):
        return course

    current_credit = course.get("credits")
    current_credit = current_credit.strip() if isinstance(current_credit, str) else None
    authoritative_code = course.get("code")
    authoritative = authoritative_lookup.get((program, plan, authoritative_code))
    if not isinstance(authoritative, dict):
        return course

    authoritative_credit = authoritative.get("credits")
    if not isinstance(authoritative_credit, str):
        return course
    authoritative_credit = authoritative_credit.strip()
    if not _AUTHORITATIVE_CREDIT_RE.fullmatch(authoritative_credit):
        return course

    source_entries = course.get("source_provenance")
    source_identity = {
        (
            entry.get("program"),
            entry.get("source_filename"),
            entry.get("source_page"),
        )
        for entry in source_entries
        if isinstance(entry, dict)
    } if isinstance(source_entries, list) else set()
    repair_key = next(
        (
            key
            for key in SOURCE_VERIFIED_DESCRIPTION_CREDIT_REPAIRS
            if key[0] == program
            and key[1] == plan
            and key[2] == authoritative_code
            and (key[0], key[3], key[4]) in source_identity
        ),
        None,
    )

    # A valid parsed credit that conflicts with authoritative data is left
    # untouched and therefore fails closed.  It must not be overwritten by a
    # source-backed exception.
    if current_credit and not _PARENTHETICAL_CREDIT_RE.fullmatch(current_credit):
        if current_credit != authoritative_credit:
            if repair_key is not None and not _INCOMPLETE_CREDIT_FRAGMENT_RE.fullmatch(
                current_credit
            ):
                return course
            if repair_key is None:
                return course
        else:
            return course

    if (
        current_credit
        and _PARENTHETICAL_CREDIT_RE.fullmatch(current_credit)
        and authoritative_credit[authoritative_credit.index("(") :] != current_credit
    ):
        return course

    if current_credit:
        repaired = dict(course)
        repaired["credits"] = authoritative_credit
        return repaired

    if repair_key is None:
        return course

    if SOURCE_VERIFIED_DESCRIPTION_CREDIT_REPAIRS[repair_key] != authoritative_code:
        return course

    repaired = dict(course)
    repaired["credits"] = authoritative_credit
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
    authoritative_lookup = _load_authoritative_credit_lookup(resolved_program, resolved_plan)

    for file in files_to_process:
        # Ignore already extracted files
        if file.name.endswith("_extracted.json"):
            continue

        print(f"\nProcessing OCR output: {file.name}")
        result = extractor.process_file(file)
        result["courses"] = [
            _reconcile_source_backed_credit(
                course,
                authoritative_lookup,
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
