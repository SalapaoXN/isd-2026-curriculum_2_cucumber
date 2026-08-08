import argparse
import json
from pathlib import Path


def sync_categories(gened_filepaths, dsba_filepath, output_filepath):
    """Sync course categories from one or more GENED source files into a DSBA file by course code."""
    # 1. Load data from all gened source files and build a mapping of course code -> category
    gened_map = {}
    for gened_filepath in gened_filepaths:
        with open(gened_filepath, "r", encoding="utf-8") as f:
            gened_data = json.load(f)

        for course in gened_data.get("courses", []):
            code = course.get("code")
            if code:
                gened_map[code] = course.get("category", "")

    # 2. Load the dsba target file
    with open(dsba_filepath, "r", encoding="utf-8") as f:
        dsba_data = json.load(f)

    # 3. Update category in the dsba file wherever the course code matches
    update_count = 0
    for course in dsba_data.get("courses", []):
        code = course.get("code")

        # If this course code exists in the gened file, replace its category
        if code in gened_map:
            course["category"] = gened_map[code]
            update_count += 1

    # 4. Save the updated data
    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dsba_data, f, ensure_ascii=False, indent=4)

    print(
        f"Done! Mapped and updated categories for {update_count} courses (from {len(gened_filepaths)} files) -> {output_path}"
    )
    return update_count


def sync_categories_by_file(gened_filepath, dsba_filepath, output_filepath):
    """Sync course categories from a single GENED source file into a DSBA file by course code."""
    return sync_categories([gened_filepath], dsba_filepath, output_filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Sync course categories from a GENED source file into a DSBA file by course code."
    )
    parser.add_argument(
        "--gened",
        required=True,
        help="GENED source JSON (source file of categories), e.g. consolidated_page_151-224.json",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="The DSBA JSON to modify, e.g. dsba_coop_full.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="New output file name (if not specified, overwrites the --input file)",
    )
    args = parser.parse_args()

    output = args.output or args.input
    sync_categories_by_file(args.gened, args.input, output)


if __name__ == "__main__":
    main()
