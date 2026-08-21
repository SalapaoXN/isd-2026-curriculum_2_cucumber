import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def extract_page_num(file_path: Path) -> int:
    match = re.search(r"page_(\d+)", file_path.name)
    return int(match.group(1)) if match else -1


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


def dedupe_courses(courses: List[dict]) -> List[dict]:
    """Preserve source entries; course codes are not unique placement keys."""
    return list(courses)


class CurriculumConsolidator:
    """Merge plan table courses with course descriptions (primary = plan)."""

    def __init__(self, plan_data: Dict, description_data: Dict):
        self.plan_data = plan_data
        self.description_data = description_data

    def consolidate(self) -> Dict:
        descriptions = self.description_data.get("descriptions") or self.description_data.get("courses", [])

        desc_lookup = {}
        for desc in descriptions:
            code = desc.get("code")
            if isinstance(code, str) and code:
                desc_lookup[code] = desc

        consolidated_courses = []
        processed_codes = set()

        for course in self.plan_data.get("courses", []):
            course_code = course.get("code")
            if not isinstance(course_code, str):
                course_code = ""
            merged_course = course.copy()

            if "หรือ" not in course_code and course_code in desc_lookup:
                target_desc = desc_lookup[course_code]
                for field in ("prerequisite", "desc_th", "desc_en"):
                    if field in target_desc:
                        merged_course[field] = target_desc[field]
                processed_codes.add(course_code)

            elif "หรือ" in course_code:
                sub_codes = [c.strip() for c in course_code.split("หรือ")]
                th_list = []
                en_list = []
                for sub_code in sub_codes:
                    if sub_code in desc_lookup:
                        target_desc = desc_lookup[sub_code]
                        if target_desc.get("desc_th"):
                            th_list.append(target_desc.get("desc_th"))
                        if target_desc.get("desc_en"):
                            en_list.append(target_desc.get("desc_en"))
                if th_list:
                    merged_course["desc_th"] = "\n".join(th_list)
                if en_list:
                    merged_course["desc_en"] = "\n".join(en_list)
                processed_codes.add(course_code)

            else:
                if course_code:
                    processed_codes.add(course_code)

            merged_course.setdefault("flexible_year_semester", None)
            consolidated_courses.append(merged_course)

        for code, desc_item in desc_lookup.items():
            if code not in processed_codes:
                new_elective_course = desc_item.copy()
                new_elective_course.setdefault("year", 0)
                new_elective_course.setdefault("semester", 0)
                new_elective_course.setdefault("category", "หมวดวิชาเฉพาะ")
                new_elective_course.setdefault("type", "เลือก")
                new_elective_course.setdefault("prerequisite", "ไม่มี")
                new_elective_course.setdefault("flexible_year_semester", None)
                consolidated_courses.append(new_elective_course)
                processed_codes.add(code)

        return {
            "source": self.plan_data.get("source", "Merged Academic Plan & Course Descriptions"),
            "description": self.plan_data.get("description", "Ground Truth รายวิชาหลักสูตร"),
            "program": self.plan_data.get("program", ""),
            "plan": self.plan_data.get("plan", ""),
            "total_courses": len(consolidated_courses),
            "courses": consolidated_courses,
        }


def merge_plan_with_description(table_courses: List[dict], desc_courses: List[dict], metadata: dict) -> dict:
    """Combine table courses with description courses."""
    plan_data = dict(metadata)
    plan_data["courses"] = table_courses
    desc_data = {"courses": desc_courses}
    return CurriculumConsolidator(plan_data, desc_data).consolidate()


def merge_consecutive_files(
    input_dir: str = "outputs",
    output_dir: str = "consolidated_outputs",
    plan_filter: str = None,
    pages: str = None,
    prefix: str = None,
    desc_pages: str = None,
):
    """Group *_extracted.json files by plan + consecutive pages, dedupe courses by code, and merge."""
    input_path = Path(input_dir)
    output_folder = Path(output_dir)
    group_id = _safe_identifier(prefix) if prefix else _input_group_identifier(input_path)

    json_files = list(input_path.glob("*_extracted.json"))
    if not json_files:
        print(f" No *_extracted.json files found in folder: {input_path.resolve()}")
        return

    target_pages = parse_page_range(pages) if pages else None
    target_desc_pages = parse_page_range(desc_pages) if desc_pages else None

    # 1. Read each file: (plan, page_num, data)
    records: List[Tuple[str, int, dict]] = []
    for file in json_files:
        if prefix and not file.name.startswith(prefix):
            continue
        page_num = extract_page_num(file)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        plan = data.get("plan", "unknown")
        is_desc_page = target_desc_pages is not None and page_num in target_desc_pages
        if plan_filter and plan != plan_filter and not is_desc_page:
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
            if not isinstance(code, str) or not code:
                continue
            known = code_lookup.get(code)
            if known is None:
                code_lookup[code] = course
            elif course.get("prerequisite") not in (None, "") and known.get("prerequisite") in (None, ""):
                code_lookup[code] = course

    def enrich_prerequisite(course: dict) -> dict:
        code = course.get("code")
        prereq = course.get("prerequisite")
        if not isinstance(code, str) or not code or prereq not in (None, ""):
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
            for plan_name, page_num, data in group:
                for course in data.get("courses", []):
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
            safe_plan = _safe_identifier(plan)
            output_filename = f"merged_{group_id}_{safe_plan}_page_{min(page_nums):03d}-{max(page_nums):03d}.json"
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

    # 5. Optional: combine each plan's table courses with its description courses into one full file.
    if desc_pages:
        target_desc_pages = parse_page_range(desc_pages)
        combined_count = 0
        for plan in plans:
            plan_records = sorted(
                (r for r in records if r[0] == plan), key=lambda r: r[1]
            )
            table_records = [r for r in plan_records if r[1] not in target_desc_pages]
            desc_records = [r for r in records if r[1] in target_desc_pages]
            if not table_records or not desc_records:
                continue

            table_courses = []
            for _, _, data in table_records:
                table_courses.extend(data.get("courses", []))
            desc_courses = []
            for _, _, data in desc_records:
                desc_courses.extend(data.get("courses", []))

            first = table_records[0][2]
            metadata = {
                "source": first.get("source", ""),
                "description": first.get("description", ""),
                "program": first.get("program", "DSBA"),
                "plan": first.get("plan", "coop"),
            }
            final = merge_plan_with_description(
                dedupe_courses(table_courses),
                dedupe_courses(desc_courses),
                metadata,
            )

            safe_plan = _safe_identifier(plan)
            output_filename = f"merged_{group_id}_{safe_plan}_full.json"
            output_file_path = output_folder / output_filename
            output_folder.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(final, f, ensure_ascii=False, indent=4)

            combined_count += 1
            print(
                f" Combined plan '{plan}' table ({len(table_records)} files) + "
                f"description ({len(desc_records)} files) "
                f"-> {len(final.get('courses', []))} courses: {output_file_path.name}"
            )

        print(f"\n Table+description merge successful! {combined_count} full file(s) saved.")


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
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Only merge files whose name starts with this prefix (e.g. 'gened')",
    )
    parser.add_argument(
        "-d",
        "--desc-pages",
        type=str,
        default=None,
        help="Page range of course descriptions to merge with the plan table (e.g. '317-344')",
    )

    args = parser.parse_args()

    merge_consecutive_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        plan_filter=args.plan,
        pages=args.p,
        prefix=args.prefix,
        desc_pages=args.desc_pages,
    )
