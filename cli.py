import argparse
from pathlib import Path
from src import OCREngine, save_ocr_results


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract text from images using EasyOCR and save to .txt and .json files."
    )
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="outputs", help="Output directory (default: 'outputs')"
    )
    parser.add_argument(
        "-l", "--languages", type=str, default="th,en", help="Comma-separated language codes (default: 'th,en')"
    )
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU mode")

    return parser.parse_args()


def main():
    args = parse_arguments()

    input_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    languages = [lang.strip() for lang in args.languages.split(",")]
    use_gpu = not args.no_gpu

    try:
        # Step 1: Initialize Engine
        engine = OCREngine(languages=languages, gpu=use_gpu)

        # Step 2: Perform Extraction
        print(f"\nProcessing: {input_path}")
        lines = engine.extract_text(input_path, detail=0)

        # Step 3: Print Results to Console
        print(f"\nFound {len(lines)} lines of text:")
        for idx, line in enumerate(lines, start=1):
            print(f" {idx:02d} | {line}")
        print("-" * 40)

        # Step 4: Export Files
        save_ocr_results(lines, output_dir, input_path.stem)

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()