import argparse
import json
from pathlib import Path
from typing import List

from .checker import OCRSpellChecker
from .english_name_enricher import enrich_courses
from .extractor import CurriculumExtractor
from .file_handler import save_ocr_results
from .ocr_engine import OCREngine
from .pre_clean import pre_clean_with_regex

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def parse_pages(pages_str: str) -> List[int]:
    """Convert a page spec string such as '32-36' or '16,17,20' into a list of integers"""
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
        help="Directory containing images (default: 'inputs/dsba')"
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
    parser.add_argument(
        "--english-second-pass",
        action="store_true",
        help="Enable opt-in English-only course-name enrichment",
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = parse_pages(args.pages)
    if not pages:
        print(" Invalid page format! Please specify e.g. -p 32-36 or -p 16,17,18")
        return

    print(" Starting the Auto OCR -> Extract Pipeline")
    print(f" Pages to process: {pages}")

    use_gpu = not args.no_gpu
    engine = OCREngine(languages=["th", "en"], gpu=use_gpu)
    english_engine = None
    if args.english_second_pass:
        if not use_gpu:
            print(" English second pass skipped because --no-gpu was requested.")
        else:
            try:
                import torch

                if not torch.cuda.is_available():
                    print(" English second pass skipped because CUDA is unavailable.")
                else:
                    english_engine = OCREngine(languages=["en"], gpu=True)
            except Exception as exc:
                print(f" English second pass skipped: {type(exc).__name__}")
    # spell_checker = OCRSpellChecker()
    extractor = CurriculumExtractor(program=args.program, plan=args.plan)

    for page_num in pages:
        base_name = f"{input_dir.name}_page_{page_num:03d}"

        img_file = None
        for ext in IMAGE_EXTENSIONS:
            candidate = input_dir / f"{base_name}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if not img_file:
            print(f"\n  [Skip] No image file found for page {page_num} in '{input_dir}'")
            continue

        print(f"\n========================================")
        print(f" Processing: {img_file.name}")
        print(f"========================================")

        # Step 1: OCR
        lines = engine.extract_text(img_file, detail=0)
        print(f"   ├─ OCR read {len(lines)} lines")

        lines = [line.upper() for line in lines]

        # Step 1.5: Deterministic regex pre-clean (before extraction)
        raw_text_joined = "\n".join(lines)
        raw_text_joined = pre_clean_with_regex(raw_text_joined)
        lines = [line for line in raw_text_joined.split("\n") if line.strip()]

        # Step 1.5: Autocorrect English text (pyspellchecker)
        # lines, typos = spell_checker.process_lines(lines)
        # print(f"   ├─ Autocorrect fixed {len(typos)} points")
        # if typos:
        #     for t in typos:
        #         print(f"   │    L{t['line']}: {t['original']} -> {t['corrected']}")

        save_ocr_results(
            lines,
            output_dir,
            base_name,
            source_filename=img_file.name,
            source_page=page_num,
            program=args.program,
        )

        # Step 2: Extract
        ocr_json_file = output_dir / f"{base_name}_ocr.json"
        extracted_data = extractor.process_file(ocr_json_file)

        # Step 2.5: Post-extraction cleaning is handled by the deterministic
        # extractor itself (pre_clean + universal anchors).  No LLM needed.
        if english_engine is not None:
            enrich_courses(extracted_data, img_file, english_engine)

        # Step 3: Save Output
        output_filename = output_dir / f"{base_name}_ocr_extracted.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)

        courses_count = len(extracted_data.get("courses", []))
        print(f"   └─  Extracted successfully ({courses_count} courses) -> saved at '{output_filename.name}'")

    print(f"\n Finished processing all pages! Files saved at: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
