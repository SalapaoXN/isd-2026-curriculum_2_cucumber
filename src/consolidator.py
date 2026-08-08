import argparse
import json
from pathlib import Path
from typing import Dict, Union


class CurriculumConsolidator:
    def __init__(self, plan_data: Dict, description_data: Dict):
        """
        :param plan_data: JSON data from the study plan table (contains a list of courses)
        :param description_data: JSON data from course description pages (contains a list of descriptions or courses)
        """
        self.plan_data = plan_data
        self.description_data = description_data

    def consolidate(self) -> Dict:
        # Support both "descriptions" or "courses" keys in the description file
        descriptions = self.description_data.get("descriptions") or self.description_data.get("courses", [])
        
        # 1. Build an Index Lookup from the course description file
        desc_lookup = {}
        for desc in descriptions:
            code = desc.get("code")
            if code:
                desc_lookup[code] = desc

        consolidated_courses = []
        processed_codes = set()  # keep track of codes already handled to avoid duplicate courses

        # 2. Loop over courses from the study plan (Plan Data) as the primary source
        for course in self.plan_data.get("courses", []):
            course_code = course.get("code", "")
            
            # Copy all properties of the original course from the plan so no field is lost
            merged_course = course.copy()

            # Case 2.1: single course code (no "หรือ")
            if "หรือ" not in course_code and course_code in desc_lookup:
                target_desc = desc_lookup[course_code]
                
                # [Fixed]: do not overwrite everything, preserve flexible_year_semester and existing data
                # update only prerequisite (and type/desc if present)
                if "prerequisite" in target_desc:
                    merged_course["prerequisite"] = target_desc["prerequisite"]
                    
                if "desc_th" in target_desc:
                    merged_course["desc_th"] = target_desc["desc_th"]
                    
                if "desc_en" in target_desc:
                    merged_course["desc_en"] = target_desc["desc_en"]

                processed_codes.add(course_code)

            # Case 2.2: paired elective code like "06026259 หรือ 06026260"
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
                        processed_codes.add(sub_code)

                if th_list:
                    merged_course["desc_th"] = "\n".join(th_list)
                if en_list:
                    merged_course["desc_en"] = "\n".join(en_list)
                processed_codes.add(course_code)

            else:
                if course_code:
                    processed_codes.add(course_code)

            # -----------------------------------------------------
            # Set defaults to prevent schema breakage if a field was never present
            # If it already exists, the value will not be changed
            merged_course.setdefault("flexible_year_semester", None)
            # -----------------------------------------------------

            consolidated_courses.append(merged_course)

        # 3. Pull courses that exist only in the course descriptions (electives/new courses) and append them
        for code, desc_item in desc_lookup.items():
            if code not in processed_codes:
                new_elective_course = desc_item.copy()
                
                # Set default values for electives if not specified
                new_elective_course.setdefault("year", 0)
                new_elective_course.setdefault("semester", 0)
                new_elective_course.setdefault("category", "หมวดวิชาเฉพาะ")
                new_elective_course.setdefault("type", "เลือก")
                new_elective_course.setdefault("prerequisite", "ไม่มี")
                
                new_elective_course.setdefault("flexible_year_semester", None)
                
                consolidated_courses.append(new_elective_course)
                processed_codes.add(code)

        # 4. Build the output data
        return {
            "source": self.plan_data.get("source", "Merged Academic Plan & Course Descriptions"),
            "description": self.plan_data.get("description", "Ground Truth รายวิชาหลักสูตร"),
            "program": self.plan_data.get("program", ""),
            "plan": self.plan_data.get("plan", ""),
            "total_courses": len(consolidated_courses),
            "courses": consolidated_courses,
        }


def save_json(data: Dict, output_path: Union[str, Path]):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f" File saved successfully: {output_path} (total {data.get('total_courses', 0)} courses)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate Plan and Description JSON files by page ranges.")
    
    # 1. Accept the page range of the study plan (Plan Pages)
    parser.add_argument(
        "-p", "--plan-pages", 
        required=True, 
        help="Page range of the study plan, e.g. 030-036 or 023-029"
    )
    
    # 2. Accept the page range of the course descriptions (Desc Pages)
    parser.add_argument(
        "-d", "--desc-pages", 
        required=True, 
        help="Page range of the course descriptions, e.g. 314-341"
    )
    
    # 3. Set the output file name
    parser.add_argument(
        "-o", "--output", 
        default=None, 
        help="Output file name or path to save to"
    )

    # 4. Specify the GENED file (e.g. consolidated_page_151-224.json) to sync GENED course categories
    #    If not specified, automatically search remaining consolidated_page_*.json files in consolidated_outputs
    parser.add_argument(
        "--gened",
        default=None,
        help="GENED consolidated JSON (e.g. consolidated_page_151-224.json) for syncing categories into the Output file",
    )

    args = parser.parse_args()

    #  Define the Base Folder
    base_dir = Path("consolidated_outputs")
    
    plan_file = base_dir / f"consolidated_page_{args.plan_pages}.json"
    desc_file = base_dir / f"consolidated_page_{args.desc_pages}.json"

    if not plan_file.exists():
        plan_file = base_dir / f"page_{args.plan_pages}.json"
    if not desc_file.exists():
        desc_file = base_dir / f"page_{args.desc_pages}.json"

    if not plan_file.exists():
        raise FileNotFoundError(f" Plan table file not found: {plan_file}")
    if not desc_file.exists():
        raise FileNotFoundError(f" Course description file not found: {desc_file}")

    print(f" Loading plan table file: {plan_file.name}")
    print(f" Loading course description file: {desc_file.name}")

    with open(plan_file, "r", encoding="utf-8") as f:
        plan_data = json.load(f)

    with open(desc_file, "r", encoding="utf-8") as f:
        desc_data = json.load(f)

    # Merge
    consolidator = CurriculumConsolidator(plan_data, desc_data)
    final_result = consolidator.consolidate()

    #  Set output path automatically if -o is not specified
    if args.output:
        output_path = Path(args.output)
        if not output_path.suffix:
            output_path = output_path / f"consolidated_page_{args.plan_pages}.json"
    else:
        output_path = base_dir / f"final_consolidated_page_{args.plan_pages}.json"

    save_json(final_result, output_path)

    # 5. Sync GENED course categories into the Output file (runs automatically)
    #    Must run after consolidation and before evaluating the file
    #    - If --gened is specified, use that file directly
    #    - If not, search the remaining consolidated_page_*.json files in base_dir as the category source
    if args.gened:
        gened_sources = [Path(args.gened)]
        if not gened_sources[0].exists():
            gened_sources[0] = base_dir / args.gened
        if not gened_sources[0].exists():
            raise FileNotFoundError(f" GENED file not found: {gened_sources[0]}")
    else:
        gened_sources = [
            f
            for f in base_dir.glob("consolidated_page_*.json")
            if f not in (plan_file, desc_file, output_path)
        ]

    if gened_sources:
        from transfer import sync_categories

        names = ", ".join(f.name for f in gened_sources)
        print(f" Syncing GENED categories from: {names}")
        sync_categories(gened_sources, output_path, output_path)
    else:
        print(" No GENED files found for category sync (skipping step)")