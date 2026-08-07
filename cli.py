import argparse
from pathlib import Path
from src import OCREngine, save_ocr_results


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract text from a single image or an entire directory of images using EasyOCR."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an image file or a directory containing images."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save OCR outputs (default: 'outputs')"
    )
    parser.add_argument(
        "-l", "--languages",
        type=str,
        default="th,en",
        help="Comma-separated language codes (default: 'th,en')"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU mode"
    )

    return parser.parse_args()


def get_image_files(input_path: Path) -> list[Path]:
    """Return a list of valid image paths from a file or folder."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [input_path]
        else:
            raise ValueError(f"File {input_path.name} is not a supported image format.")

    elif input_path.is_dir():
        # Search for all supported image extensions inside folder
        image_files = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        # Sort files alphabetically (e.g., page_001.jpg, page_002.jpg)
        image_files.sort(key=lambda x: x.name)
        return image_files

    return []


def main():
    args = parse_arguments()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    languages = [lang.strip() for lang in args.languages.split(",")]
    use_gpu = not args.no_gpu

    try:
        # Step 1: Discover Image Files
        images = get_image_files(input_path)
        if not images:
            print(f"❌ No supported image files found in {input_path}")
            return

        print(f"🔍 Found {len(images)} image(s) to process.")

        # Step 2: Initialize OCR Engine ONCE for all images
        engine = OCREngine(languages=languages, gpu=use_gpu)

        # Step 3: Loop Through All Images
        for idx, img_file in enumerate(images, start=1):
            print(f"\n========================================")
            print(f"[{idx}/{len(images)}] Processing image: {img_file.name}")
            print(f"========================================")

            lines = engine.extract_text(img_file, detail=0)

            # Print quick preview to console
            print(f"Found {len(lines)} line(s) of text.")

            # Step 4: Save Individual OCR Results (.txt and .json)
            save_ocr_results(lines, output_dir, img_file.stem)

        print(f"\n✅ Finished processing all {len(images)} image(s)!")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()