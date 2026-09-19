"""Single end-to-end pipeline entry point (clean rebuild).

Stages:
  ocr -> extract -> merge -> correct -> (evaluate) -> (build_index)

Final RAG-ready output: data/output/final/*_corrected.json (+ optional curriculum.db).
scripts/ask.py stays separate and is not run here.

Defaults (adjust only by flags, behavior unchanged):
  - evaluate.py IS included by default; pass --skip-eval to skip it.
  - build_index is OFF by default; pass --with-index to continue to
    curriculum.db, or --only-index to build only the DB from existing
    corrected files.

Intermediates (extracted/, page_ranges/, legacy_summaries/) go to a temp
directory and are deleted automatically. Pass --keep-intermediates for
debug to persist them under the output tree.

Resume points (--from):
  ocr          run everything from OCR (default)
  extracted    skip OCR and per-file extraction; merge from existing extracted/
  consolidated skip to LLM correction from existing consolidated full files
  corrected    skip to evaluate/index from existing corrected files
Resume modes read persisted data/output/ layout, so they pair with
--keep-intermediates output from a previous debug run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_PROGRAMS = ("ait", "bit", "dsba", "gened", "it")
FROM_CHOICES = ("ocr", "extracted", "consolidated", "corrected")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR -> extract -> merge -> correct -> (evaluate) -> (build_index).",
        epilog="Example: python -m src.pipeline.run --program it --with-index",
    )
    parser.add_argument("--program", choices=SUPPORTED_PROGRAMS, default="it")
    parser.add_argument("--pages", default=None, help="OCR pages/ranges, e.g. 32-38 or 32-34,328-330.")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluate.py stage.")
    parser.add_argument("--with-index", action="store_true", help="Continue to curriculum.db after correction.")
    parser.add_argument("--only-index", action="store_true", help="Only build curriculum.db from existing corrected files.")
    parser.add_argument("--from", dest="from_stage", choices=FROM_CHOICES, default="ocr")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _program_scoped_prepare(program_key: str, ocr_root: Path, extracted_root: Path, consolidated_root: Path) -> dict:
    """Run the prepare_data scope logic for one program only (no duplication)."""
    from src.pipeline.tools.preparation import tool as _pd

    key = program_key.casefold()
    if not _pd.prepare_program(key, ocr_root / key, extracted_root, consolidated_root, BASE_DIR):
        raise _pd.PreparationError(f"No usable OCR source for program '{key}' under {ocr_root}")
    return {"prepared_programs": [key]}


def _copy_final_full_files(consolidated_tmp: Path, consolidated_final: Path) -> list[Path]:
    kept: list[Path] = []
    for src in sorted(consolidated_tmp.glob("**/full/merged_*_full.json")):
        rel = src.relative_to(consolidated_tmp)
        dest = consolidated_final / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        kept.append(dest)
    return kept


def _corrected_paths_for_program(llm_dir: Path, program_key: str) -> list[Path]:
    wanted = program_key.upper()
    kept: list[Path] = []
    for path in sorted(llm_dir.glob("*_corrected.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and str(doc.get("program", "")).upper() == wanted:
            kept.append(path)
    return kept


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("Stages: ocr -> extract -> merge -> correct -> (evaluate) -> (build_index)")
    print(f"Program={args.program} from={args.from_stage} keep_intermediates={args.keep_intermediates}")
    if args.dry_run:
        print("Dry run: no stages executed.")
        return 0
    if args.only_index and args.from_stage != "ocr":
        print("Note: --only-index ignores --from; building index only.", file=sys.stderr)

    output_base = Path(args.output_dir)
    if not output_base.is_absolute():
        output_base = BASE_DIR / output_base
    ocr_root = output_base / "ocr"
    extracted_final = output_base / "extracted"
    consolidated_final = output_base / "consolidated"
    llm_dir = output_base / "final"

    if args.only_index:
        from src.pipeline.tools.indexing.tool import run_build_index_stage

        paths = _corrected_paths_for_program(llm_dir, args.program) or None
        print(run_build_index_stage(paths, args.index_path))
        return 0

    from_stage = args.from_stage
    do_ocr = from_stage == "ocr"
    do_extract_merge = from_stage in ("ocr", "extracted")
    do_correct = from_stage in ("ocr", "extracted", "consolidated")

    tmp_holder: tempfile.TemporaryDirectory | None = None
    try:
        if args.keep_intermediates:
            extracted_root = extracted_final
            consolidated_work = consolidated_final
        else:
            tmp_holder = tempfile.TemporaryDirectory(prefix="pipeline_")
            tmp_base = Path(tmp_holder.name)
            extracted_root = tmp_base / "extracted"
            consolidated_work = tmp_base / "consolidated"

        if do_ocr:
            from src.pipeline.tools.ocr.tool import run_ocr_stage

            run_ocr_stage(args.program, args.pages, args.no_gpu, BASE_DIR)

        if do_extract_merge:
            if from_stage == "extracted":
                # Merge only, from persisted extracted/ (debug resume path).
                from src.pipeline.tools.preparation import tool as _pd

                config = _pd.PROGRAM_CONFIG[args.program.casefold()]
                for scope in config.scopes:
                    _pd._run_merge(
                        extracted_final / args.program.casefold(),
                        consolidated_work,
                        config,
                        scope,
                        _pd._combined_pages(scope, True),
                    )
            else:
                _program_scoped_prepare(args.program, ocr_root, extracted_root, consolidated_work)
            if not args.keep_intermediates:
                kept = _copy_final_full_files(consolidated_work, consolidated_final)
                print(f"Kept {len(kept)} final full file(s) under {consolidated_final}")

        if do_correct:
            from src.pipeline.tools.correction.tool import run_correct_stage

            if from_stage == "consolidated":
                inputs: list[Path] | None = None
            else:
                full_files = sorted(consolidated_final.glob("**/full/merged_*_full.json"))
                wanted = [p for p in full_files if args.program.casefold() in p.as_posix().casefold()]
                inputs = wanted or None
            if inputs is None:
                corrected = run_correct_stage(consolidated_final, llm_dir)
            else:
                corrected = run_correct_stage(consolidated_final, llm_dir, inputs)
            print(f"Corrected {len(corrected)} file(s) under {llm_dir}")

        if not args.skip_eval:
            from src.pipeline.tools.evaluation.tool import run_evaluate_stage

            program_paths = _corrected_paths_for_program(llm_dir, args.program)
            if program_paths:
                from src.pipeline.tools.evaluation.evaluate import discover_llm_evaluation_pairs

                all_pairs = discover_llm_evaluation_pairs(llm_dir)
                pairs = [(p, g) for p, g in all_pairs if Path(p) in set(program_paths)]
                run_evaluate_stage(pairs=pairs)
            else:
                print(f"No corrected files for {args.program}; skipping evaluation.")

        if args.with_index:
            from src.pipeline.tools.indexing.tool import run_build_index_stage

            program_paths = _corrected_paths_for_program(llm_dir, args.program)
            print(run_build_index_stage(program_paths or None, args.index_path))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if tmp_holder is not None:
            tmp_holder.cleanup()
    print(f"Done. RAG-ready files under {llm_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
