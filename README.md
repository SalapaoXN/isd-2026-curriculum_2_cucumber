# CUCUMBER

**P2 ระบบถาม-ตอบข้อมูลหลักสูตรด้วย LLM**

CUCUMBER เป็นระบบถาม-ตอบข้อมูลหลักสูตรภาษาไทย โดยให้ **LLM ช่วยตีความว่าผู้ใช้ต้องการถามอะไร** แต่ข้อเท็จจริง เช่น รหัสวิชา หน่วยกิต ปี/เทอม แผนการเรียน และวิชาบังคับก่อน จะต้องมาจาก **ฐานข้อมูล SQLite และหลักฐานที่ระบบตรวจสอบแล้ว** เท่านั้น

ถ้าระบบระบุวิชา ขอบเขตหลักสูตร หรือหลักฐานได้ไม่ชัดเจน ระบบจะ **หยุดและขอข้อมูลเพิ่มหรือไม่ตอบ** แทนการคาดเดา (fail closed)

สมาชิก:

1. 67070049 Nattachai Kaewchum — Discord: GoodDee
2. 67070063 Thanachin Chukiatchai — Discord: วันลพ มีงบมาก
3. 67070103 Pongsakorn Panyacom — Discord: เบบี๋คือดวงใจ

## 1. Set up the environment

Python **3.10–3.13** is supported (`pyproject.toml` declares the range); Python 3.11 is
recommended for broad OCR/PyTorch wheel availability. Node.js 20+ is required. Python
version constraints belong in project metadata, not as a `pip install` requirement.

Create the virtual environment using the commands for your platform.

**Windows PowerShell:**

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd ci --prefix frontend
```

Use a Python version from **3.10 to 3.13**. If `python` is not recognized, install Python
and enable **Add python.exe to PATH**, then reopen PowerShell. The Windows `py` launcher is
optional; it is not installed with every Python distribution.

**macOS / Linux:**

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
npm ci --prefix frontend
```

The `torch` and `torchvision` requirements are intentionally unpinned so pip selects
compatible wheels for the operating system. NVIDIA GPU OCR may require installing a
CUDA-enabled PyTorch build for that machine; `--no-gpu` provides a CPU fallback.

Create a root `.env` file and set your key:

```text
GEMINI_API_KEY=your-gemini-api-key
```

## 2. Run OCR - Build-Index

Run GENED first in each era because program merges backfill empty course descriptions
from the matching GENED final (`gened2564` for the 2565/2566 era;
`gened2557` for 2560):

**Windows PowerShell:**

```powershell
$datasets = @(
    "gened2564", "ait2566", "bit2565", "dsba2565", "it2565",
    "gened2557", "bit2560", "dsba2560", "it2560"
)

foreach ($dataset in $datasets) {
    & .\.venv\Scripts\python.exe -m src.pipeline.run --dataset $dataset --skip-eval
    if ($LASTEXITCODE -ne 0) { throw "Pipeline failed for $dataset" }
}

& .\.venv\Scripts\python.exe -m src.pipeline.run_rules
& .\.venv\Scripts\python.exe -m src.pipeline.tools.evaluation.evaluate --reports-dir reports/evaluation
& .\.venv\Scripts\python.exe -m src.pipeline.tools.canonicalize_runtime
& .\.venv\Scripts\python.exe -m src.pipeline.tools.preflight_runtime
& .\.venv\Scripts\python.exe -m rag.build_index
```

**macOS / Linux:**

```sh
datasets=(gened2564 ait2566 bit2565 dsba2565 it2565 gened2557 bit2560 dsba2560 it2560)
for dataset in "${datasets[@]}"; do
    .venv/bin/python -m src.pipeline.run --dataset "$dataset" --skip-eval || exit 1
done

.venv/bin/python -m src.pipeline.run_rules
.venv/bin/python -m src.pipeline.tools.evaluation.evaluate --reports-dir reports/evaluation
.venv/bin/python -m src.pipeline.tools.canonicalize_runtime
.venv/bin/python -m src.pipeline.tools.preflight_runtime
.venv/bin/python -m rag.build_index
```

## 3. Build and run locally

**Windows PowerShell:**

```powershell
npm.cmd run build --prefix frontend
& .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**macOS / Linux:**

```sh
npm run build --prefix frontend
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open:

- Chat: `http://127.0.0.1:8000/`
- Curriculum browser: `http://127.0.0.1:8000/curriculum`
- Health/index status: `http://127.0.0.1:8000/api/health`
