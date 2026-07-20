import argparse
import json
from pathlib import Path
from rich import print
from .config import OCRConfig
from .pipeline import run_ocr
from .evaluation import evaluate_extracted_from_files, evaluate_from_files
from .field_extraction import extract_common_fields
from .utils.io import save_json
from .curriculum_extraction import extract_curriculum_from_file


def parse_args():
    parser = argparse.ArgumentParser(description="Thai-English OCR system")
    sub = parser.add_subparsers(dest="command", required=True)

    ocr = sub.add_parser("ocr", help="Run OCR on image or PDF")
    ocr.add_argument("input_path")
    ocr.add_argument("--output-dir", default="outputs")
    ocr.add_argument("--engine", choices=["paddle", "tesseract", "trocr", "ensemble"], default="ensemble")
    ocr.add_argument("--languages", default="tha+eng", help="Tesseract languages, e.g. tha+eng")
    ocr.add_argument("--paddle-lang", default="th", help="PaddleOCR language, e.g. th or en")
    ocr.add_argument("--dpi", type=int, default=300)
    ocr.add_argument("--no-preprocess", action="store_true")
    ocr.add_argument("--no-deskew", action="store_true")
    ocr.add_argument("--save-debug-images", action="store_true")
    ocr.add_argument("--min-confidence", type=float, default=0.0)
    ocr.add_argument("--device", default="cpu")

    curr = sub.add_parser("extract-curriculum", help="Extract curriculum data from an OCR file")
    curr.add_argument("ocr_path", help="Path to the OCR output file")
    curr.add_argument("--template-path", default=None, help="Optional path to a curriculum template file")
    curr.add_argument("--program", default="DSBA", help="Program code (default: DSBA)")
    curr.add_argument("--plan", default="no_coop", help="Plan type (default: no_coop)")
    curr.add_argument("--output-dir", default="outputs")
    

    ev = sub.add_parser("evaluate", help="Evaluate OCR JSON against ground truth JSON")
    ev.add_argument("ground_truth_json")
    ev.add_argument("prediction_json")
    ev.add_argument("--output", default="outputs/evaluation_result.json")
    ev.add_argument("--mode", choices=["ocr", "ocr-extracted"], default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "ocr":
        output_dir = Path(args.output_dir)
        config = OCRConfig(
            input_path=Path(args.input_path),
            output_dir=output_dir,
            page_image_dir=output_dir / "pages",
            engine=args.engine,
            languages=args.languages,
            paddle_lang=args.paddle_lang,
            dpi=args.dpi,
            preprocess=not args.no_preprocess,
            deskew=not args.no_deskew,
            save_debug_images=args.save_debug_images,
            min_confidence=args.min_confidence,
            device=args.device,
        )
        result = run_ocr(config)
        fields = extract_common_fields(result.text)
        field_path = output_dir / f"{Path(args.input_path).stem}_fields.json"
        save_json(fields, field_path)
        print(f"[green]OCR done[/green]: {output_dir}")
        print(f"Extracted fields: {json.dumps(fields, ensure_ascii=False, indent=2)}")

    elif args.command == "extract-curriculum":
        output_dir = Path(args.output_dir)
        result = extract_curriculum_from_file(
            ocr_path=args.ocr_path,
            template_path=args.template_path,
            program=args.program,
            plan=args.plan,
        )
        extracted_path = output_dir / f"{Path(args.ocr_path).stem}_extracted.json"
        print("[green]Curriculum extraction complete[/green]")
        save_json(result, extracted_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "evaluate":
        prediction_name = Path(args.prediction_json).name.lower()
        if args.mode == "ocr-extracted" or "extracted" in prediction_name or "curriculum" in prediction_name:
            result = evaluate_extracted_from_files(args.ground_truth_json, args.prediction_json)
        else:
            result = evaluate_from_files(args.ground_truth_json, args.prediction_json)

        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
