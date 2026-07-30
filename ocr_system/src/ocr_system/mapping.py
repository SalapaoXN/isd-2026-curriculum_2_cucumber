import argparse
import csv
import json
import re
from pathlib import Path


CODE_RE = re.compile(r"\d{5,8}")
PAGE_RE = re.compile(r"page[_-]?(0*)(\d+)", re.IGNORECASE)


def extract_codes(text):
    if not text:
        return []
    return CODE_RE.findall(str(text))


def collect_gt_codes(gt_root: Path):
    """Return dict code -> list of (gt_file, program, plan)"""
    gt_map = {}
    for p in gt_root.rglob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        program = data.get("program") or data.get("source") or ""
        plan = data.get("plan") or ""

        courses = data.get("courses") or []
        for c in courses:
            code_field = c.get("code") if isinstance(c, dict) else None
            if not code_field:
                continue
            codes = extract_codes(code_field)
            for code in codes:
                gt_map.setdefault(code, []).append((str(p.relative_to(gt_root.parent)), program, plan))

    return gt_map


def collect_ocr_pages(outputs_root: Path):
    """Return dict code -> list of (page, ocr_file)"""
    found = {}
    for p in outputs_root.rglob("*_ocr_extracted.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        m = PAGE_RE.search(p.name)
        page = m.group(2) if m else ""

        courses = data.get("courses") or []
        for c in courses:
            code_field = c.get("code") if isinstance(c, dict) else None
            if not code_field:
                continue
            codes = extract_codes(code_field)
            for code in codes:
                found.setdefault(code, []).append((page, str(p.relative_to(outputs_root.parent))))

    return found


def write_csv(out_path: Path, gt_map, ocr_map):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["code", "ground_truth_files", "ocr_pages", "ocr_files"])

        all_codes = set(gt_map.keys()) | set(ocr_map.keys())
        for code in sorted(all_codes):
            gt_entries = gt_map.get(code, [])
            gt_files = ";".join([f"{f}|{prog}|{plan}" for (f, prog, plan) in gt_entries])

            ocr_entries = ocr_map.get(code, [])
            pages = ";".join([p for (p, f) in ocr_entries])
            ocr_files = ";".join([f for (p, f) in ocr_entries])

            writer.writerow([code, gt_files, pages, ocr_files])


def main():
    parser = argparse.ArgumentParser(description="Map ground-truth course codes to OCR pages/files")
    parser.add_argument("--gt-dir", default="data/ground_truth", help="ground truth root dir")
    parser.add_argument("--outputs-dir", default="outputs", help="OCR outputs dir")
    parser.add_argument("--out-csv", default="outputs/code_page_mapping.csv", help="output CSV file")
    args = parser.parse_args()

    gt_root = Path(args.gt_dir).resolve()
    outputs_root = Path(args.outputs_dir).resolve()
    out_csv = Path(args.out_csv).resolve()

    gt_map = collect_gt_codes(gt_root)
    ocr_map = collect_ocr_pages(outputs_root)

    write_csv(out_csv, gt_map, ocr_map)
    print(f"Wrote mapping CSV: {out_csv}")


if __name__ == "__main__":
    main()
