import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from src.extractor import (
    DEFAULT_COOP_PAIRS,
    CurriculumExtractor,
    merge_alternative_courses,
    merge_source_provenance,
    parse_document_page,
    prediction_description,
    prediction_source_label,
)
from src.pre_clean import pre_clean_with_regex
from src.pipeline_config import plan_label

def _recover_credit_from_matching_description(
    plan_credit: object,
    desc_credit: object,
) -> str | None:
    if not isinstance(plan_credit, str):
        return None
    if not isinstance(desc_credit, str):
        return None

    plan_credit = plan_credit.replace(" ", "").strip()
    desc_credit = desc_credit.replace(" ", "").strip()

    # Plan lost only the leading credit:
    # "(0-3-2)"
    plan_match = re.fullmatch(
        r"(\([0-9xX]+-[0-9xX]+-[0-9xX]+\))",
        plan_credit,
    )
    if not plan_match:
        return None

    # Description still has the complete value:
    # "1(0-3-2)"
    desc_match = re.fullmatch(
        r"[0-9]+(\([0-9xX]+-[0-9xX]+-[0-9xX]+\))",
        desc_credit,
    )
    if not desc_match:
        return None

    # Recover only when the inside workload tuple is identical.
    if plan_match.group(1) != desc_match.group(1):
        return None

    return desc_credit

def _apply_gened_audit_credit(course: dict, audit_codes: set[str]) -> dict:
    code = course.get("code")
    credits = course.get("credits")

    if code not in audit_codes:
        return course

    if not isinstance(credits, str):
        return course

    credits = credits.strip()

    # Recover only a missing leading credit value.
    # Example: "(4-0-8)" + explicit Audit evidence -> "0(4-0-8)"
    if not re.fullmatch(
        r"\([0-9xX]+-[0-9xX]+-[0-9xX]+\)",
        credits,
    ):
        return course

    result = dict(course)
    result["credits"] = f"0{credits}"
    return result

def _recover_gened_structural_credit(
    course: dict,
    audit_codes: set[str],
) -> dict:
    # Do not infer anything when Audit context is unavailable.
    # This prevents a partial merge from turning an unknown
    # "(4-0-8)" Audit course into "4(4-0-8)".
    if not audit_codes:
        return course

    code = course.get("code")
    if code in audit_codes:
        return course

    # Structural recovery is for catalog/plan records only.
    provenance = course.get("source_provenance", [])
    if not any(
        isinstance(entry, dict)
        and entry.get("document_category") == "plan"
        for entry in provenance
    ):
        return course

    credits = course.get("credits")
    if not isinstance(credits, str):
        return course

    compact = credits.replace(" ", "").strip()

    # Missing outer credit only:
    # (4-0-8)
    match = re.fullmatch(
        r"\((\d+)-0-(\d+)\)",
        compact,
    )
    if not match:
        return course

    lecture_hours = int(match.group(1))
    self_study_hours = int(match.group(2))

    # Recover only the standard lecture pattern:
    # n-0-2n  ->  n(n-0-2n)
    if (
        lecture_hours <= 0
        or self_study_hours != lecture_hours * 2
    ):
        return course

    result = dict(course)
    result["credits"] = (
        f"{lecture_hours}"
        f"({lecture_hours}-0-{self_study_hours})"
    )
    return result

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


def _prediction_metadata(data: dict, plan) -> dict:
    """Keep generated merge metadata neutral even when inputs are stale."""
    program = data.get("program", "")
    source = data.get("source")
    description = data.get("description")
    if not isinstance(source, str) or not source.strip() or "gt_template" in source.casefold() or "ground truth" in source.casefold():
        source = prediction_source_label(program, plan)
    if not isinstance(description, str) or not description.strip() or "ground truth" in description.casefold():
        description = prediction_description(program, plan)
    return {
        "source": source,
        "description": description,
        "program": program,
        "plan": plan,
    }


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


def _raw_ocr_path(extracted_path: Path) -> Path | None:
    if extracted_path.name.endswith("_ocr_extracted.json"):
        json_path = extracted_path.with_name(
            extracted_path.name.replace("_ocr_extracted.json", "_ocr.json")
        )
        if json_path.exists():
            return json_path
        txt_path = extracted_path.with_name(
            extracted_path.name.replace("_ocr_extracted.json", "_ocr.txt")
        )
        if txt_path.exists():
            return txt_path
    return None


def _read_raw_ocr(raw_path: Path) -> tuple[List[str], dict]:
    if raw_path.suffix.lower() == ".json":
        with raw_path.open("r", encoding="utf-8") as handle:
            raw_data = json.load(handle)
        if not isinstance(raw_data, dict):
            return [], {}
        return raw_data.get("text_lines", []), raw_data
    return raw_path.read_text(encoding="utf-8").splitlines(), {}


def _hydrate_page_document_pages(
    data: dict, extracted_path: Path, plan: str
) -> None:
    """Fill missing page metadata from stored OCR without changing provenance identity."""
    raw_path = _raw_ocr_path(extracted_path)
    if raw_path is None:
        return

    lines, metadata = _read_raw_ocr(raw_path)
    categories = {
        entry.get("document_category")
        for course in data.get("courses", [])
        for entry in course.get("source_provenance", [])
        if isinstance(entry, dict)
    }
    categories.intersection_update({"plan", "description"})
    if not categories:
        return

    extractor = CurriculumExtractor(
        program=data.get("program", "DSBA"),
        plan=plan,
    )
    for category in categories:
        source_context = extractor._source_context(
            raw_path,
            metadata,
            category,
            text_lines=lines,
        )
        document_page = source_context.get("document_page")
        source_page = source_context.get("source_page")
        if document_page is None:
            continue

        for course in data.get("courses", []):
            for entry in course.get("source_provenance", []):
                if (
                    not isinstance(entry, dict)
                    or entry.get("document_category") != category
                    or entry.get("source_page") != source_page
                    or parse_document_page(entry.get("document_page")) is not None
                ):
                    continue
                entry["document_page"] = document_page


def _load_ordered_description_courses(
    page_records: List[Tuple[str, int, dict]], source_files: Dict[Tuple[str, int], Path]
) -> List[dict] | None:
    """Re-extract stored description OCR pages in numeric order when available."""
    if not page_records:
        return []

    first_plan, _, first_data = page_records[0]
    extractor = CurriculumExtractor(
        program=first_data.get("program", "DSBA"),
        plan=first_plan,
    )
    pages = []

    for plan, page_num, page_data in sorted(page_records, key=lambda item: item[1]):
        extracted_path = source_files.get((plan, page_num))
        raw_path = _raw_ocr_path(extracted_path) if extracted_path else None
        if raw_path is None:
            return None

        lines, metadata = _read_raw_ocr(raw_path)

        normalized_lines = [str(line).upper() for line in lines]
        cleaned = pre_clean_with_regex("\n".join(normalized_lines))
        normalized_lines = [line for line in cleaned.split("\n") if line.strip()]
        raw_source_context = extractor._source_context(
            raw_path, metadata, "description", text_lines=lines
        )
        source_context = raw_source_context
        if raw_path.suffix.lower() != ".json":
            for course in page_data.get("courses", []):
                provenance = course.get("source_provenance", [])
                description_entries = [
                    entry
                    for entry in provenance
                    if isinstance(entry, dict)
                    and entry.get("document_category") == "description"
                ]
                if description_entries:
                    source_context = dict(description_entries[0])
                    if parse_document_page(source_context.get("document_page")) is None:
                        source_context["document_page"] = raw_source_context.get(
                            "document_page"
                        )
                    break
        pages.append((normalized_lines, source_context))

    return extractor.extract_descriptions_from_pages(pages).get("courses", [])


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
        gened_catalog_merge = self.plan_data.get("program") == "GENED"

        desc_lookup = {}
        desc_occurrences: Dict[str, List[dict]] = {}
        for desc in descriptions:
            code = desc.get("code")
            if isinstance(code, str) and code:
                desc_lookup[code] = desc
                desc_occurrences.setdefault(code, []).append(desc)

        plan_occurrences: Dict[str, int] = {}
        for course in self.plan_data.get("courses", []):
            course_code = course.get("code")
            if not isinstance(course_code, str) or not course_code:
                continue
            codes = (
                [part.strip() for part in course_code.split("หรือ")]
                if "หรือ" in course_code
                else [course_code]
            )
            for code in codes:
                if code:
                    plan_occurrences[code] = plan_occurrences.get(code, 0) + 1

        def unique_description(code: str) -> dict | None:
            occurrences = desc_occurrences.get(code, [])
            if len(occurrences) == 1:
                return occurrences[0]
            return None

        def has_ambiguous_description(code: str) -> bool:
            return len(desc_occurrences.get(code, [])) > 1

        consolidated_courses = []
        processed_codes = set()
        ambiguous_codes = set()
        for code, occurrences in desc_occurrences.items():
            if len(occurrences) > 1 and plan_occurrences.get(code, 0) == 0:
                ambiguous_codes.add(code)

        for course in self.plan_data.get("courses", []):
            course_code = course.get("code")
            if not isinstance(course_code, str):
                course_code = ""
            merged_course = course.copy()

            if "หรือ" not in course_code and course_code in desc_lookup:
                target_desc = unique_description(course_code)

                if target_desc is None:
                    if has_ambiguous_description(course_code):
                        ambiguous_codes.add(course_code)

                else:
                    recovered_credit = _recover_credit_from_matching_description(
                        merged_course.get("credits"),
                        target_desc.get("credits"),
                    )

                    if recovered_credit is not None:
                        merged_course["credits"] = recovered_credit

                    for field in ("prerequisite", "desc_th", "desc_en"):
                        if field in target_desc:
                            merged_course[field] = target_desc[field]

                    merged_course["source_provenance"] = merge_source_provenance(
                        course,
                        target_desc,
                    )

                processed_codes.add(course_code)

            elif "หรือ" in course_code:
                sub_codes = [c.strip() for c in course_code.split("หรือ")]
                th_list = []
                en_list = []
                description_sources = []
                for sub_code in sub_codes:
                    target_desc = unique_description(sub_code)
                    if target_desc is not None:
                        if target_desc.get("desc_th"):
                            th_list.append(target_desc.get("desc_th"))
                        if target_desc.get("desc_en"):
                            en_list.append(target_desc.get("desc_en"))
                        description_sources.append(target_desc)
                    elif has_ambiguous_description(sub_code):
                        ambiguous_codes.add(sub_code)
                if th_list:
                    merged_course["desc_th"] = "\n".join(th_list)
                if en_list:
                    merged_course["desc_en"] = "\n".join(en_list)
                merged_course["source_provenance"] = merge_source_provenance(
                    course, *description_sources
                )
                processed_codes.add(course_code)
                if self.plan_data.get("program") in {"IT", "BIT"}:
                    processed_codes.update(sub_codes)

            else:
                if course_code:
                    processed_codes.add(course_code)

            merged_course.setdefault("flexible_year_semester", None)
            consolidated_courses.append(merged_course)

        # IT's and BIT's co-op alternatives are present in the description catalog even
        # when the no_coop plan table has no corresponding plan row.  Merge
        # only this configured pair when both descriptions are unique and
        # neither code has a plan occurrence.  Other curricula keep their
        # existing unmatched-description behavior.
        description_pair_by_code = {}
        description_pair_heads = set()
        description_pair = {
            "IT": ("06016481", "06016482"),
            "BIT": ("06036147", "06036148"),
        }.get(self.plan_data.get("program"))
        if description_pair:
            for code_a, code_b, credits in DEFAULT_COOP_PAIRS:
                if (code_a, code_b) != description_pair:
                    continue
                first_description = desc_occurrences.get(code_a, [])
                second_description = desc_occurrences.get(code_b, [])
                if (
                    plan_occurrences.get(code_a, 0) == 0
                    and plan_occurrences.get(code_b, 0) == 0
                    and len(first_description) == 1
                    and len(second_description) == 1
                ):
                    merged_description = merge_alternative_courses(
                        first_description[0], second_description[0], credits
                    )
                    description_pair_by_code[code_a] = merged_description
                    description_pair_by_code[code_b] = merged_description
                    description_pair_heads.add(code_a)

        for code, desc_item in desc_lookup.items():
            if code in description_pair_by_code:
                if code in description_pair_heads:
                    consolidated_courses.append(description_pair_by_code[code])
                    processed_codes.update(description_pair_by_code[code]["code"].split(" หรือ "))
                continue
            if code not in processed_codes and code not in ambiguous_codes:
                if gened_catalog_merge:
                    continue
                new_elective_course = desc_item.copy()
                new_elective_course.setdefault("year", 0)
                new_elective_course.setdefault("semester", 0)
                new_elective_course.setdefault("category", "หมวดวิชาเฉพาะ")
                new_elective_course.setdefault("type", "เลือก")
                new_elective_course.setdefault("prerequisite", "ไม่มี")
                new_elective_course.setdefault("flexible_year_semester", None)
                consolidated_courses.append(new_elective_course)
                processed_codes.add(code)

        unresolved_descriptions = [
            desc.copy()
            for desc in descriptions
            if isinstance(desc.get("code"), str) and desc.get("code") in ambiguous_codes
        ]
        if gened_catalog_merge:
            unresolved_descriptions.extend(
                desc.copy()
                for desc in descriptions
                if (
                    isinstance(desc.get("code"), str)
                    and desc.get("code") not in plan_occurrences
                    and desc.get("code") not in ambiguous_codes
                )
            )

        result = {
            **_prediction_metadata(self.plan_data, self.plan_data.get("plan")),
            "total_courses": len(consolidated_courses),
            "courses": consolidated_courses,
        }
        if unresolved_descriptions:
            result["unresolved_descriptions"] = unresolved_descriptions
        return result


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
    source_files: Dict[Tuple[str, int], Path] = {}
    for file in json_files:
        if prefix and not file.name.startswith(prefix):
            continue
        page_num = extract_page_num(file)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        artifact_plan = data.get("plan", "unknown")
        is_desc_page = target_desc_pages is not None and page_num in target_desc_pages
        plan = plan_filter if is_desc_page and plan_filter is not None else artifact_plan
        if plan_filter and plan != plan_filter:
            continue
        if target_pages is not None and page_num not in target_pages:
            continue
        _hydrate_page_document_pages(data, file, plan)
        records.append((plan, page_num, data))
        source_files[(plan, page_num)] = file
    
    audit_codes = {
        code
        for _, _, data in records
        for code in data.get("audit_course_codes", [])
        if isinstance(code, str) and code
    }

    if audit_codes:
        for _, _, data in records:
            if data.get("program") != "GENED":
                continue

            data["courses"] = [
                _recover_gened_structural_credit(
                    _apply_gened_audit_credit(
                        course,
                        audit_codes,
                    ),
                    audit_codes,
                )
                for course in data.get("courses", [])
            ]

    # 2. Build a code -> course lookup from ALL files (Study Plan + Course Description
    #    pages together) so prerequisites can be enriched even across separate groups.
    code_lookup: Dict[str, dict] = {}
    plan_occurrences: Dict[str, int] = {}
    description_occurrences: Dict[str, int] = {}
    for _, _, data in records:
        for course in data.get("courses", []):
            code = course.get("code")
            if not isinstance(code, str) or not code:
                continue
            categories = {
                entry.get("document_category")
                for entry in course.get("source_provenance", [])
                if isinstance(entry, dict)
            }
            if "plan" in categories:
                plan_occurrences[code] = plan_occurrences.get(code, 0) + 1
            if "description" in categories:
                description_occurrences[code] = description_occurrences.get(code, 0) + 1
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
        if code in plan_occurrences or code in description_occurrences:
            if plan_occurrences.get(code, 0) != 1 or description_occurrences.get(code, 0) != 1:
                return course
        merged_course = dict(course)
        for field in ("prerequisite", "desc_th", "desc_en"):
            if field in desc_course and desc_course.get(field) not in (None, ""):
                merged_course[field] = desc_course[field]
        merged_course["source_provenance"] = merge_source_provenance(
            course, desc_course
        )
        return merged_course

    # 3. Group by plan
    plans = sorted({r[0] for r in records}, key=lambda value: "" if value is None else str(value))
    ordered_description_courses: Dict[str, List[dict] | None] = {}
    if target_desc_pages is not None:
        for plan in plans:
            description_page_records = [
                record
                for record in records
                if record[0] == plan and record[1] in target_desc_pages
            ]
            ordered_description_courses[plan] = _load_ordered_description_courses(
                description_page_records, source_files
            )
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
            group_has_description_pages = (
                target_desc_pages is not None
                and any(page_num in target_desc_pages for _, page_num, _ in group)
            )
            for plan_name, page_num, data in group:
                if group_has_description_pages and page_num in target_desc_pages:
                    continue
                for course in data.get("courses", []):
                    all_courses.append(enrich_prerequisite(course))

            if group_has_description_pages:
                all_courses.extend(ordered_description_courses.get(plan) or [])

            first = group[0][2]
            base_metadata = _prediction_metadata(first, plan)
            base_metadata["total_courses"] = len(all_courses)
            base_metadata["courses"] = all_courses

            page_nums = [r[1] for r in group]
            safe_plan = _safe_identifier(plan_label(plan))
            output_filename = f"merged_{group_id}_{safe_plan}_page_{min(page_nums):03d}-{max(page_nums):03d}.json"
            output_file_path = output_folder / output_filename

            output_folder.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(base_metadata, f, ensure_ascii=False, indent=4)

            merged_count += 1
            print(
                f" Merged {len(group)} files (plan '{plan_label(plan)}', "
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
            desc_records = [r for r in plan_records if r[1] in target_desc_pages]
            if not table_records or not desc_records:
                continue

            table_courses = []
            for _, _, data in table_records:
                table_courses.extend(data.get("courses", []))
            desc_courses = ordered_description_courses.get(plan)
            if desc_courses is None:
                desc_courses = []
                for _, _, data in desc_records:
                    desc_courses.extend(data.get("courses", []))

            first = table_records[0][2]
            table_courses = CurriculumExtractor(
                program=first.get("program", "DSBA"),
                plan=plan,
            ).post_process(table_courses)
            metadata = _prediction_metadata(first, plan)
            final = merge_plan_with_description(
                dedupe_courses(table_courses),
                dedupe_courses(desc_courses),
                metadata,
            )

            safe_plan = _safe_identifier(plan_label(plan))
            output_filename = f"merged_{group_id}_{safe_plan}_full.json"
            output_file_path = output_folder / output_filename
            output_folder.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(final, f, ensure_ascii=False, indent=4)

            combined_count += 1
            print(
                f" Combined plan '{plan_label(plan)}' table ({len(table_records)} files) + "
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
