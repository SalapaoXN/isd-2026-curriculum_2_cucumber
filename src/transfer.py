import argparse
import json
from pathlib import Path


def sync_categories(gened_filepaths, dsba_filepath, output_filepath):
    """Sync course categories from one or more GENED source files into a DSBA file by course code."""
    # 1. โหลดข้อมูลจากไฟล์ gened (ไฟล์ต้นทาง) ทุกไฟล์ แล้วรวมเป็น Mapping รหัสวิชา -> หมวดหมู่
    gened_map = {}
    for gened_filepath in gened_filepaths:
        with open(gened_filepath, "r", encoding="utf-8") as f:
            gened_data = json.load(f)

        for course in gened_data.get("courses", []):
            code = course.get("code")
            if code:
                gened_map[code] = course.get("category", "")

    # 2. โหลดข้อมูลจากไฟล์ dsba (ไฟล์ปลายทาง)
    with open(dsba_filepath, "r", encoding="utf-8") as f:
        dsba_data = json.load(f)

    # 3. อัปเดต category ในไฟล์ dsba ทันทีที่รหัสวิชา (code) ตรงกัน
    update_count = 0
    for course in dsba_data.get("courses", []):
        code = course.get("code")

        # ถ้ารหัสวิชานี้มีอยู่ในไฟล์ gened ให้ทำการแทนที่ category
        if code in gened_map:
            course["category"] = gened_map[code]
            update_count += 1

    # 4. บันทึกข้อมูลที่อัปเดตแล้ว
    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dsba_data, f, ensure_ascii=False, indent=4)

    print(
        f"ดำเนินการเสร็จสิ้น! ทำการแมพและอัปเดตหมวดหมู่ไปทั้งหมด {update_count} รายวิชา (จาก {len(gened_filepaths)} ไฟล์) -> {output_path}"
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
        help="GENED source JSON (ไฟล์ต้นทางของหมวดหมู่) เช่น consolidated_page_151-224.json",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="DSBA JSON ที่ต้องการแก้ เช่น dsba_coop_full.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="ชื่อไฟล์เอาต์พุตใหม่ (ถ้าไม่ระบุจะเขียนทับไฟล์ --input)",
    )
    args = parser.parse_args()

    output = args.output or args.input
    sync_categories_by_file(args.gened, args.input, output)


if __name__ == "__main__":
    main()
