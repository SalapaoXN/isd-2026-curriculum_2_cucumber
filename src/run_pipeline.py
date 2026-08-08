import argparse
import json
from pathlib import Path
from typing import List

from .checker import OCRSpellChecker
from .extractor import CurriculumExtractor
from .file_handler import save_ocr_results
from .ocr_engine import OCREngine

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def parse_pages(pages_str: str) -> List[int]:
    """แปลงข้อความรับระบุหน้า เช่น '32-36' หรือ '16,17,20' ให้เป็น List ตัวเลข"""
    pages = set()
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            pages.update(range(int(start), int(end) + 1))
        elif ".." in part:
            start, end = part.split("..")
            pages.update(range(int(start), int(end) + 1))
        elif part.isdigit():
            pages.add(int(part))
    return sorted(list(pages))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Auto OCR -> Extraction Pipeline for specific pages."
    )
    parser.add_argument(
        "-p", "--pages",
        type=str,
        required=True,
        help="Specify pages or ranges (e.g. '32-36', '16,17,20', '16')"
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default=str(BASE_DIR / "inputs/dsba"),
        help="Directory containing images (default: 'inputs')"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=str(BASE_DIR / "outputs"),
        help="Directory to save outputs (default: 'outputs')"
    )
    parser.add_argument(
        "--program",
        type=str,
        default="DSBA",
        help="Program name (default: 'DSBA')"
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="coop",
        help="Study plan name (default: 'coop')"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU mode"
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = parse_pages(args.pages)
    if not pages:
        print(" รูปแบบเลขหน้าไม่ถูกต้อง! กรุณาระบุ เช่น -p 32-36 หรือ -p 16,17,18")
        return

    print(" เริ่มต้นระบบ Auto OCR -> Extract Pipeline")
    print(f" หน้าที่จะประมวลผล: {pages}")

    use_gpu = not args.no_gpu
    engine = OCREngine(languages=["th", "en"], gpu=use_gpu)
    spell_checker = OCRSpellChecker()
    extractor = CurriculumExtractor(program=args.program, plan=args.plan)

    for page_num in pages:
        base_name = f"curriculum_page_{page_num:03d}"

        img_file = None
        for ext in IMAGE_EXTENSIONS:
            candidate = input_dir / f"{base_name}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if not img_file:
            print(f"\n  [Skip] ไม่พบไฟล์รูปภาพสำหรับหน้า {page_num} ใน '{input_dir}'")
            continue

        print(f"\n========================================")
        print(f" Processing: {img_file.name}")
        print(f"========================================")

        # Step 1: OCR
        lines = engine.extract_text(img_file, detail=0)
        print(f"   ├─ OCR อ่านได้ {len(lines)} บรรทัด")

        lines = [line.upper() for line in lines]

        # Step 1.5: Autocorrect English text (pyspellchecker)
        lines, typos = spell_checker.process_lines(lines)
        print(f"   ├─ Autocorrect แก้ไข {len(typos)} จุด")
        if typos:
            for t in typos:
                print(f"   │    L{t['line']}: {t['original']} -> {t['corrected']}")

        save_ocr_results(lines, output_dir, base_name)

        # Step 2: Extract
        ocr_json_file = output_dir / f"{base_name}_ocr.json"
        extracted_data = extractor.process_file(ocr_json_file)

        # Step 3: Save Output
        output_filename = output_dir / f"{base_name}_ocr_extracted.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)

        courses_count = len(extracted_data.get("courses", []))
        print(f"   └─  Extracted สำเร็จ (พบ {courses_count} วิชา) -> บันทึกที่ '{output_filename.name}'")

    print(f"\n เสร็จเรียบร้อยทุกหน้า! บันทึกไฟล์ไว้ที่: {output_dir.resolve()}")


if __name__ == "__main__":
    main()