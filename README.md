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

`python -m src.run_pipeline` checks exact page filenames in `--input-dir` and does not search nested directories. `python cli.py` accepts one image or scans images directly inside a directory; its filenames do not need the automated page naming convention.

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

`src.run_pipeline` accepts `-p/--pages`, `-i/--input-dir`, `-o/--output-dir`, `--program`, `--plan`, `--no-gpu`, and `--english-second-pass`. It performs OCR, extraction, and output writing for the requested pages.

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

### Testing

Tracked regression suites cover evaluator behavior, English enrichment, and source provenance:
```bash
python -m unittest discover -s tests -p "test_evaluate.py"
python -m unittest discover -s tests -p "test_english_name_enricher.py"
python -m unittest discover -s tests -p "test_provenance.py"
```

### Experimental and current scope

The active repository is primarily a curriculum data-preparation pipeline: OCR, structured extraction, optional English name enrichment, source/page provenance, consolidation, and evaluation. The normal OCR path needs no OpenAI API key, `openai` package, or Ollama service. The LLM cleaner under `src/llm_clean_txt.py` is inactive and not required by the production OCR path.

Dedicated Rules extraction, RAG/retrieval, chatbot, API, database, and UI implementations are not complete production components in this repository.
