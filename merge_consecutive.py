import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def extract_page_num(file_path: Path) -> int:
    match = re.search(r"page_(\d+)", file_path.name)
    return int(match.group(1)) if match else -1


def parse_page_range(page_input: str) -> set:
    pages = set()
    parts = page_input.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            pages.update(range(int(start), int(end) + 1))
        elif part.isdigit():
            pages.add(int(part))
    return pages


def merge_consecutive_files(
    input_dir: str = "outputs",
    output_dir: str = "consolidated_outputs",
    plan_filter: str = None,
    pages: str = None,
):
    """Group *_extracted.json files by plan + consecutive pages, dedupe courses by code, and merge."""
    input_path = Path(input_dir)
    output_folder = Path(output_dir)

    json_files = list(input_path.glob("*_extracted.json"))
    if not json_files:
        print(f" No *_extracted.json files found in folder: {input_path.resolve()}")
        return

    target_pages = parse_page_range(pages) if pages else None

    # 1. Read each file: (plan, page_num, data)
    records: List[Tuple[str, int, dict]] = []
    for file in json_files:
        page_num = extract_page_num(file)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        plan = data.get("plan", "unknown")
        if plan_filter and plan != plan_filter:
            continue
        if target_pages is not None and page_num not in target_pages:
            continue
        records.append((plan, page_num, data))

    # 2. Build a code -> course lookup from ALL files (Study Plan + Course Description
    #    pages together) so prerequisites can be enriched even across separate groups.
    code_lookup: Dict[str, dict] = {}
    for _, _, data in records:
        for course in data.get("courses", []):
            code = course.get("code")
            if code:
                code_lookup.setdefault(code, course)

    def enrich_prerequisite(course: dict) -> dict:
        code = course.get("code")
        prereq = course.get("prerequisite")
        if not code or prereq not in (None, ""):
            return course
        desc_course = code_lookup.get(code)
        if not desc_course:
            return course
        merged_course = dict(course)
        for field in ("prerequisite", "desc_th", "desc_en"):
            if field in desc_course and desc_course.get(field) not in (None, ""):
                merged_course[field] = desc_course[field]
        return merged_course

    # 3. Group by plan
    plans = sorted({r[0] for r in records})
    merged_count = 0

    for plan in plans:
        plan_records = sorted(
            (r for r in records if r[0] == plan), key=lambda r: r[1]
        )

        # 3. Split into consecutive page groups
        groups: List[List[Tuple[str, int, dict]]] = []
        current: List[Tuple[str, int, dict]] = []
        for rec in plan_records:
            if current and rec[1] != current[-1][1] + 1:
                groups.append(current)
                current = []
            current.append(rec)
        if current:
            groups.append(current)

        # 4. Merge each consecutive group
        for group in groups:
            all_courses = []
            seen_codes = set()
            for plan_name, page_num, data in group:
                for course in data.get("courses", []):
                    code = course.get("code")
                    if code in seen_codes:
                        continue
                    seen_codes.add(code)
                    all_courses.append(enrich_prerequisite(course))

            first = group[0][2]
            base_metadata = {
                "source": first.get("source", ""),
                "description": first.get("description", ""),
                "program": first.get("program", "DSBA"),
                "plan": first.get("plan", "coop"),
            }
            base_metadata["total_courses"] = len(all_courses)
            base_metadata["courses"] = all_courses

            page_nums = [r[1] for r in group]
            output_filename = f"merged_{plan}_page_{min(page_nums):03d}-{max(page_nums):03d}.json"
            output_file_path = output_folder / output_filename

            output_folder.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(base_metadata, f, ensure_ascii=False, indent=4)

            merged_count += 1
            print(
                f" Merged {len(group)} files (plan '{plan}', "
                f"pages {min(page_nums):03d}-{max(page_nums):03d}) "
                f"-> {len(all_courses)} courses: {output_file_path.name}"
            )

    print(f"\n Merge successful! {merged_count} output file(s) saved in {output_folder.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge OCR JSON files grouped by plan and consecutive pages"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="outputs",
        help="Input folder containing *_extracted.json",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="consolidated_outputs",
        help="Output folder for merged JSON",
    )
    parser.add_argument(
        "--plan",
        type=str,
        default=None,
        help="Only merge files with this plan (e.g. 'gened')",
    )
    parser.add_argument(
        "-p",
        type=str,
        default=None,
        help="Only merge these page numbers (e.g. '16-18', '16,18')",
    )

    args = parser.parse_args()

    merge_consecutive_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        plan_filter=args.plan,
        pages=args.p,
    )
