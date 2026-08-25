# isd-2026-curriculum_2_cucumber
We do OCR curriculum and some LLM with model name CUCUMBER

Project : P2 LLM ถาม-ตอบหลักสูตร

Member:
1. 67070049 Nattachai Kaewchum >> Discord: GoodDee
2. 67070063 Thanachin Chukiatchai >> Discord: วันลพ มีงบมาก
3. 67070103 Pongsakorn Panyacom >> Discord: เบบี๋คือดวงใจ


## Setup

Use Python `3.10–3.13`. Python `3.11` is the preferred baseline.

From a fresh clone, create and activate a virtual environment from the repository root:

**Windows PowerShell**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Unix-like (macOS/Linux)**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the pinned dependencies:
```bash
python -m pip install -r requirements.txt
```

The baseline does not configure CUDA. Use `--no-gpu` for CPU execution; GPU support is optional and follows the existing EasyOCR/runtime environment.

### EasyOCR models

The canonical OCR engine uses EasyOCR with Thai and English (`['th', 'en']`). On first use, EasyOCR may download missing detector and recognition models; later runs reuse its local cache.

Use `--no-gpu` for the reproducible CPU baseline. To check the canonical models in CPU mode:
```bash
python -c "import easyocr; easyocr.Reader(['th', 'en'], gpu=False); print('EasyOCR models ready')"
```

### Inputs and outputs

Input images are not included. The automated production runner expects the input directory name, a zero-padded page number, and a supported extension, for example:
```text
inputs/dsba/dsba_page_026.jpg
```

Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp`.

`python -m src.run_pipeline` checks direct-child page filenames in `--input-dir` and does not search nested directories. When `--pages` is omitted, it discovers files matching `<input-directory-name>_page_<NNN>.<ext>`, deduplicates page numbers, and sorts them numerically. `python cli.py` accepts one image or scans images directly inside a directory; its filenames do not need the automated page naming convention.

The pipeline creates `outputs/` and `consolidated_outputs/` when needed. Automated runs write per-page OCR text/JSON and extracted JSON files such as:
```text
outputs/dsba_page_026_ocr.txt
outputs/dsba_page_026_ocr.json
outputs/dsba_page_026_ocr_extracted.json
```

### Pipeline overview

The current production flow is:
```text
Image
  -> canonical EasyOCR (`['th', 'en']`)
  -> structured course extraction
  -> optional English name enrichment
  -> merge/consolidation
  -> evaluation
```

The canonical Thai/English OCR and extracted fields are authoritative by default. English enrichment is opt-in and changes only `name_en` when a safe auxiliary candidate is accepted.

### Canonical production run

Run the normal canonical path with the reproducible CPU baseline:
```bash
python -m src.run_pipeline -p 26 -i inputs/dsba -o outputs/smoke --program DSBA --plan coop --no-gpu
```

`src.run_pipeline` accepts optional `-p/--pages`, `-i/--input-dir`, `-o/--output-dir`, `--program`, `--plan`, `--no-gpu`, and `--english-second-pass`. If pages are omitted, the runner discovers direct-child page images. The program is derived only from supported input directory names (`dsba`, `it`, `ait`, `gened`, `bit`) when `--program` is omitted. DSBA, IT, and BIT require an explicit `--plan coop` or `--plan no_coop`; GENED requires `--plan gened`; AIT has no plan and must omit `--plan`.

For a one-page smoke check, provide `inputs/dsba/dsba_page_026.jpg` yourself and verify the three files listed above after the command completes. This checks execution and output creation only; it does not set an OCR accuracy threshold. Do not commit the image or generated smoke outputs.

### Optional English name enrichment

The second pass is opt-in only. The default pipeline remains canonical-only:
```bash
python -m src.run_pipeline -p 26 -i inputs/dsba -o outputs/enriched --program DSBA --plan coop --english-second-pass
```

When enabled, an auxiliary English-only OCR pass may improve `name_en`. Unsafe or ambiguous candidates fall back to the canonical `name_en`; Thai names, credits, prerequisites, categories, and other canonical fields are not replaced. The canonical `['th', 'en']` OCR remains authoritative.

The decision and auxiliary evidence are stored per record under `english_second_pass`, separately from `source_provenance`. `--no-gpu` skips the auxiliary pass. The pass also safely skips when CUDA is unavailable or the auxiliary engine cannot be initialized, while the canonical pipeline continues.

Do not assume this pass runs by default or that every candidate is accepted. `experiments/dsba_english_second_pass.py` remains an experiment, not the production path.

### Batch OCR, extraction, and merge

For a separate OCR-only batch, use the generic CLI:
```bash
python cli.py inputs/dsba/ -o outputs --no-gpu
python extract.py outputs/ -o outputs --program DSBA --plan coop
```

`extract.py` also accepts one `.txt`/`.json` OCR file or a directory. When matching `.txt` and `.json` files share a stem, JSON is preferred and the page is processed once. It writes per-file extracted JSON and, for multiple inputs, a consolidated summary.

Merge extracted plan and description pages with:
```bash
python merge_consecutive.py --input-dir outputs --output-dir consolidated_outputs --prefix dsba --plan coop -d 317-344
```

Page-range outputs use `merged_<group>_<plan>_page_<start>-<end>.json`; full table-plus-description outputs use `merged_<group>_<plan>_full.json`.

### Source provenance

Each extracted record has the additive `source_provenance` field:
```json
{
  "source_provenance": [
    {
      "program": "DSBA",
      "source_filename": "dsba_page_026.png",
      "source_page": 26,
      "document_category": "plan"
    }
  ]
}
```

Provenance is derived from source/input context, not ground truth. It survives extraction and merge; repeated records retain independent entries, and merged plan/description records may contain multiple entries. `document_category` is `plan`, `description`, or `unknown` where the source context cannot establish it. This field is separate from `english_second_pass`.

The automated runner writes the original image filename, page, and program into OCR metadata. Legacy or explicitly supplied TXT/JSON inputs use safe filename/page fallbacks when that metadata is unavailable.

### Evaluation

Evaluate a consolidated file against accepted ground truth:
```bash
python evaluate.py consolidated_outputs/merged_dsba_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_coop.json
python evaluate.py consolidated_outputs/merged_dsba_no_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_no_coop.json
```

The evaluator preserves its existing text metrics and output levels:
- CER and WER for matched text fields
- legacy `field_level`, `page_level`, and `category_level`

It also reports coverage separately:
- GT record count
- prediction record count
- matched, missing, and extra records
- precision, recall, and F1

The additive `rubric` namespace reports:
- `rubric.overall_text`
- `rubric.field_level`, including text quality and field-presence coverage
- `rubric.page_level`, with true per-page metrics only when every GT record has authoritative `source_provenance`
- `rubric.category_level`, grouped by GT curriculum `category`

Current DSBA ground truth does not contain authoritative per-record page provenance, so rubric Page Level reports `unavailable` rather than inferring pages. `code_page_mapping.csv` is a project-created helper and is not authoritative ground truth. Legacy evaluation outputs remain available for compatibility.

## Usage / Command Reference

### OCR CLI

`cli.py` runs EasyOCR only. It accepts one image or images directly inside a directory:
```bash
# Single image
python cli.py inputs/dsba/curriculum_page_016.jpg

# Directory batch
python cli.py inputs/dsba/

# Custom output, languages, and CPU mode
python cli.py inputs/dsba/ -o outputs/dsba_raw_ocr --languages th,en --no-gpu
```

### Extraction

`extract.py` accepts an OCR TXT file, OCR JSON file, or directory:
```bash
# Explicit TXT input
python extract.py outputs/curriculum_page_016_ocr.txt --program DSBA --plan coop

# Explicit JSON input
python extract.py outputs/curriculum_page_016_ocr.json --program DSBA --plan coop

# Directory input
python extract.py outputs/ --program DSBA --plan coop
python extract.py outputs/dsba_raw_ocr --program DSBA --plan coop

# Explicit metadata and output directory
python extract.py outputs/curriculum_page_016_ocr.txt \
  --program DSBA \
  --plan coop \
  --output-dir outputs/extracted \
  --source "GT_Template-2.xlsx / Academic Plan GT — DSBA coop"
```

### Automated page runner

The automated runner supports `-i/--input-dir`, `-o/--output-dir`, `--program`, `--plan`, `--no-gpu`, and the opt-in `--english-second-pass`. `-p/--pages` is optional; omission discovers direct-child files using the input directory's page filename convention. Do not use an inferred co-op variant: DSBA, IT, and BIT require explicit `coop` or `no_coop`.
```bash
# DSBA
python -m src.run_pipeline -p 26-32 -i inputs/dsba --plan no_coop --program DSBA
python -m src.run_pipeline -p 33-39 -i inputs/dsba --plan coop --program DSBA
python -m src.run_pipeline -p 317-344 -i inputs/dsba --plan coop --program DSBA

# IT
python -m src.run_pipeline -p 32-38 -i inputs/it --plan no_coop --program IT
python -m src.run_pipeline -p 39-45 -i inputs/it --plan coop --program IT
python -m src.run_pipeline -p 328-371 -i inputs/it --plan coop --program IT
# BIT
# Supply BIT source images and confirmed plan page ranges before running.

# AIT: no --plan; omitted pages are discovered automatically
python -m src.run_pipeline -i inputs/ait --program AIT

# General education: explicit GENED program and plan
python -m src.run_pipeline -i inputs/gened --plan gened --program GENED
```

### Consolidation

`merge_consecutive.py` supports `--input-dir`, `--output-dir`, `--prefix`, `--plan`, `-p/--pages`, and `-d/--desc-pages`:
```bash
# DSBA
python merge_consecutive.py --prefix dsba --plan coop -d 317-344
python merge_consecutive.py --prefix dsba --plan no_coop -d 317-344

# IT and BIT description pages, after the corresponding source inputs are available
python merge_consecutive.py --prefix it --plan coop -d 328-371
python merge_consecutive.py --prefix it --plan no_coop -d 328-371
python merge_consecutive.py --prefix bit --plan coop -d 328-371
python merge_consecutive.py --prefix bit --plan no_coop -d 328-371

# AIT has JSON plan null and uses a neutral filename label only
python merge_consecutive.py --prefix ait -d 287-302
python merge_consecutive.py --prefix gened
```

Use explicit directories when outputs are not in the defaults:
```bash
python merge_consecutive.py \
  --input-dir outputs \
  --output-dir consolidated_outputs \
  --prefix dsba \
  --plan coop \
  --desc-pages 317-344
```

### LLM spell corrector
```bash
python llm_spell_corrector.py ./consolidated_outputs/merged_dsba_coop_full.json
```

### Evaluation commands

```bash
# DSBA
python evaluate.py consolidated_outputs/merged_dsba_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_coop.json
python evaluate.py consolidated_outputs/merged_dsba_no_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_no_coop.json

# IT
python evaluate.py consolidated_outputs/merged_it_coop_full.json --gt ground_truth/IT/IT_academic_plan_coop.json
python evaluate.py consolidated_outputs/merged_it_no_coop_full.json --gt ground_truth/IT/IT_academic_plan_no_coop.json

# AIT
python evaluate.py consolidated_outputs/merged_ait_no_plan_full.json --gt ground_truth/AIT/AIT_academic_plan.json

# GENED
python evaluate.py consolidated_outputs/merged_gened_gened_page_016-030.json --gt ground_truth/general_education_ground_truth.json

# Save a JSON report
python evaluate.py \
  consolidated_outputs/merged_dsba_coop_full.json \
  --gt ground_truth/DSBA/DSBA_academic_plan_coop.json \
  --out reports/dsba_coop_evaluation.json

# Batch evaluation; repeat --pair for each prediction/ground-truth pair
python evaluate.py \
  --pair consolidated_outputs/merged_dsba_coop_full.json ground_truth/DSBA/DSBA_academic_plan_coop.json \
  --pair consolidated_outputs/merged_it_coop_full.json ground_truth/IT/IT_academic_plan_coop.json
```

Every single or batch evaluation also writes these flat reports to `reports/evaluation/`:
- `evaluation.json` — existing evaluator results combined under `results`.
- `evaluation_summary.csv` — TP/FN/FP plus Precision/Recall/F1 and percentages.
- `field_metrics.csv` — CER/WER plus character and word accuracy percentages; Thai WER uses PyThaiNLP.
- `evaluation_errors.csv` — concrete GT/prediction differences for matched, missing, and extra records.

True negatives are not fabricated because this is extraction coverage, not binary classification. Use the positional prediction path with `--gt` for a single dataset, or repeat `--pair PREDICTION_JSON GROUND_TRUTH_JSON` for batch evaluation. `--out` remains optional for saving the single or combined JSON output separately.

### Testing

Tracked regression suites cover evaluator behavior, English enrichment, and source provenance:
```bash
python -m unittest discover -s tests -p "test_cli_semantics.py"
python -m unittest discover -s tests -p "test_evaluate.py"
python -m unittest discover -s tests -p "test_english_name_enricher.py"
python -m unittest discover -s tests -p "test_provenance.py"
```

### Experimental and current scope

The active repository is primarily a curriculum data-preparation pipeline: OCR, structured extraction, optional English name enrichment, source/page provenance, consolidation, and evaluation. The normal OCR path needs no OpenAI API key, `openai` package, or Ollama service. The LLM cleaner under `src/llm_clean_txt.py` is inactive and not required by the production OCR path.

Dedicated Rules extraction, RAG/retrieval, chatbot, API, database, and UI implementations are not complete production components in this repository.
