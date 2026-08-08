import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Union


def parse_page_range(page_input: Union[str, List[int], None]) -> Optional[set]:
    """Convert a page spec into a set of integers for filtering"""
    if not page_input:
        return None

    if isinstance(page_input, (list, tuple, set)):
        return set(int(p) for p in page_input)

    if isinstance(page_input, str):
        pages = set()
        # Split the text by comma, e.g. "30-32, 35"
        parts = page_input.split(",")
        for part in parts:
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                pages.update(range(int(start), int(end) + 1))
            elif part.isdigit():
                pages.add(int(part))
        return pages

    return None


def merge_curriculum_files(
    input_dir: str = "outputs",
    output_dir: str = "consolidated_outputs",
    pages: Union[str, List[int], None] = None,
    output_filename: Optional[str] = None,
):
    """Merge *_extracted.json files, optionally selecting only specified page numbers"""
    input_path = Path(input_dir)
    output_folder = Path(output_dir)

    target_pages = parse_page_range(pages)

    # 1. Find all files
    json_files = list(input_path.glob("*_extracted.json"))
    if not json_files:
        print(
            f" No *_extracted.json files found in folder: {input_path.resolve()}"
        )
        return

    def extract_page_num(file_path: Path) -> int:
        match = re.search(r"page_(\d+)", file_path.name)
        return int(match.group(1)) if match else -1

    # 2. Filter only files whose page numbers match the specified pages
    filtered_files = []
    for f in json_files:
        p_num = extract_page_num(f)
        if target_pages is None or p_num in target_pages:
            filtered_files.append((p_num, f))

    if not filtered_files:
        print(f" No files found matching the specified pages: {pages}")
        return

    # 3. Sort by page number
    filtered_files.sort(key=lambda x: x[0])

    all_courses = []
    base_metadata = {}
    included_page_nums = [p for p, _ in filtered_files if p != -1]

    # 4. Generate an output file name automatically if none is given (e.g. consolidated_page_030-036.json)
    if not output_filename:
        if included_page_nums and target_pages:
            min_p, max_p = min(included_page_nums), max(included_page_nums)
            output_filename = f"consolidated_page_{min_p:03d}-{max_p:03d}.json"
        else:
            output_filename = "consolidated_extracted.json"

    output_file_path = output_folder / output_filename

    print(
        f" Merging {len(filtered_files)} files from folder '{input_dir}'..."
    )

    # 5. Loop through and read data
    for page_num, file in filtered_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not base_metadata:
            base_metadata = {
                "source": data.get("source", ""),
                "description": data.get("description", ""),
                "program": data.get("program", "DSBA"),
                "plan": data.get("plan", "coop"),
            }

        courses = data.get("courses", [])
        all_courses.extend(courses)
        page_label = f"page {page_num:03d}" if page_num != -1 else file.name
        print(f"   ├─ [{page_label}] merged {file.name}: added {len(courses)} courses")

    consolidated_data = {
        "source": base_metadata.get("source"),
        "description": base_metadata.get("description"),
        "program": base_metadata.get("program"),
        "plan": base_metadata.get("plan"),
        "total_courses": len(all_courses),
        "courses": all_courses,
    }

    # 6. Save the file
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(consolidated_data, f, ensure_ascii=False, indent=4)

    print(
        f"\n Merge successful! Total {len(all_courses)} courses"
    )
    print(f" Output file saved at: {output_file_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge selected OCR JSON files by page range"
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
        help="Output folder for consolidated JSON",
    )
    parser.add_argument(
        "-p",
        "--pages",
        type=str,
        default=None,
        help="Page selection (e.g. '30-36', '30,31,35', '30-32,35-36')",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="Custom output filename",
    )

    args = parser.parse_args()

    merge_curriculum_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pages=args.pages,
        output_filename=args.filename,
    )