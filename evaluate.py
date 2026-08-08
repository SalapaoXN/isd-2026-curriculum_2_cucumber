import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Optional


def string_similarity(s1: str, s2: str) -> float:
  """คำนวณความเหมือนของข้อความ (0.0 - 1.0)"""
  if not s1 and not s2:
    return 1.0
  if not s1 or not s2:
    return 0.0
  from difflib import SequenceMatcher

  return SequenceMatcher(
      None, str(s1).lower().strip(), str(s2).lower().strip()
  ).ratio()


def resolve_gt_path(pred_data: dict, gt_input: Path) -> Path:
  """ค้นหาไฟล์ Ground Truth อัตโนมัติจากโครงสร้างโฟลเดอร์โดยใช้ Metadata (program, plan)"""
  if gt_input.is_file():
    return gt_input

  # ดึง Metadata จาก Prediction JSON
  program = str(pred_data.get("program", "DSBA")).strip().upper()
  plan = str(pred_data.get("plan", "coop")).strip().lower()

  # ค้นหาตามโครงสร้าง: ground_truth/{program}/{program}_academic_plan_{plan}.json
  expected_filename = f"{program}_academic_plan_{plan}.json"
  target_path = gt_input / program / expected_filename

  if target_path.exists():
    print(f"🎯 Auto-detected Ground Truth File: {target_path}")
    return target_path

  # กรณีค้นหาแบบยืดหยุ่นในโฟลเดอร์ย่อย
  found_files = list(gt_input.glob(f"**/{expected_filename}"))
  if found_files:
    print(f"🎯 Auto-detected Ground Truth File: {found_files[0]}")
    return found_files[0]

  raise FileNotFoundError(
      f"❌ ไม่พบไฟล์ Ground Truth สำหรับ Program: '{program}', Plan: '{plan}'"
      f" ในโฟลเดอร์ {gt_input}"
  )


def compare_courses(gt_course: Dict, pred_course: Dict) -> Dict[str, bool]:
  """เปรียบเทียบข้อมูลระดับ Field ระหว่าง Ground Truth กับ Prediction"""
  fields_to_check = [
      "code",
      "name_th",
      "name_en",
      "credits",
      "year",
      "semester",
      "category",
      "type",
      "prerequisite",
  ]
  results = {}

  for field in fields_to_check:
    gt_val = gt_course.get(field, "")
    pred_val = pred_course.get(field, "")

    # สำหรับ Text ใช้ Fuzzy Match (> 0.85 ถือว่าผ่าน)
    if field in ["name_th", "name_en", "prerequisite"]:
      sim = string_similarity(gt_val, pred_val)
      results[field] = sim >= 0.85
    else:
      results[field] = str(gt_val).strip().lower() == str(
          pred_val
      ).strip().lower()

  return results


def run_evaluation(pred_path: Path, gt_input: Path):
  # 1. อ่านไฟล์ Prediction
  if not pred_path.exists():
    print(f"❌ ไม่พบไฟล์ Prediction: {pred_path}")
    return

  with open(pred_path, "r", encoding="utf-8") as f:
    pred_data = json.load(f)

  # 2. ค้นหาไฟล์ Ground Truth (ระบุตรงๆ หรือค้นหาอัตโนมัติจากโฟลเดอร์)
  try:
    gt_path = resolve_gt_path(pred_data, gt_input)
  except FileNotFoundError as e:
    print(e)
    return

  with open(gt_path, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

  gt_courses = gt_data.get("courses", [])
  pred_courses = pred_data.get("courses", [])

  # 3. จัดกลุ่ม Prediction ตาม Code (รองรับวิชาที่มีรหัสซ้ำกัน เช่น 06026xxx)
  pred_map = defaultdict(list)
  for c in pred_courses:
    code = str(c.get("code", "")).strip().lower()
    if code:
      pred_map[code].append(c)

  # Collection ตัวสถิติ
  field_stats = defaultdict(lambda: {"correct": 0, "total": 0})
  page_stats = defaultdict(
      lambda: defaultdict(lambda: {"correct": 0, "total": 0})
  )
  category_stats = defaultdict(
      lambda: defaultdict(lambda: {"correct": 0, "total": 0})
  )

  matched_count = 0

  for gt in gt_courses:
    code = str(gt.get("code", "")).strip().lower()

    # สร้าง Label ระบุหน้า/ภาคเรียน
    page = gt.get("page")
    if not page:
      y = gt.get("year", "?")
      s = gt.get("semester", "?")
      page = f"ปี {y} ภาคการศึกษาที่ {s}"

    category = gt.get("category", "ไม่ระบุหมวด")

    pred_list = pred_map.get(code, [])
    pred = pred_list.pop(0) if pred_list else None

    if pred:
      matched_count += 1
      field_results = compare_courses(gt, pred)

      for field, is_correct in field_results.items():
        field_stats[field]["total"] += 1
        if is_correct:
          field_stats[field]["correct"] += 1

        page_stats[page][field]["total"] += 1
        if is_correct:
          page_stats[page][field]["correct"] += 1

        category_stats[category][field]["total"] += 1
        if is_correct:
          category_stats[category][field]["correct"] += 1
    else:
      for field in [
          "code",
          "name_th",
          "name_en",
          "credits",
          "year",
          "semester",
          "category",
          "type",
          "prerequisite",
      ]:
        field_stats[field]["total"] += 1
        page_stats[page][field]["total"] += 1
        category_stats[category][field]["total"] += 1

  # --- DISPLAY RESULTS ---
  print("\n" + "=" * 65)
  print("📊 SUMMARY EVALUATION REPORT")
  print("=" * 65)
  print(f"🔹 Prediction File            : {pred_path.name}")
  print(f"🔹 Ground Truth File           : {gt_path.name}")
  print(f"🔹 Total Ground Truth Courses  : {len(gt_courses)}")
  print(f"🔹 Total Extracted Courses     : {len(pred_courses)}")
  print(f"🔹 Matched Courses (by Code)   : {matched_count}")

  # 1. Field-Level Summary
  print("\n" + "-" * 65)
  print("📌 1. FIELD-LEVEL ACCURACY")
  print("-" * 65)
  print(
      f"{'Field':<15} | {'Correct':<8} | {'Total':<8} | {'Accuracy (%)':<12}"
  )
  print("-" * 65)
  for field, stat in field_stats.items():
    acc = (stat["correct"] / stat["total"] * 100) if stat["total"] > 0 else 0
    print(f"{field:<15} | {stat['correct']:<8} | {stat['total']:<8} | {acc:.2f}%")

  # 2. Page / Semester Summary
  print("\n" + "-" * 65)
  print("📄 2. PAGE / SEMESTER-LEVEL ACCURACY BREAKDOWN")
  print("-" * 65)
  for page in sorted(page_stats.keys()):
    fields = page_stats[page]
    total_fields = sum(s["total"] for s in fields.values())
    correct_fields = sum(s["correct"] for s in fields.values())
    page_acc = (
        (correct_fields / total_fields * 100) if total_fields > 0 else 0
    )

    y_acc = (
        (fields["year"]["correct"] / fields["year"]["total"] * 100)
        if fields["year"]["total"] > 0
        else 0
    )
    s_acc = (
        (fields["semester"]["correct"] / fields["semester"]["total"] * 100)
        if fields["semester"]["total"] > 0
        else 0
    )

    print(
        f"► {page:<22} | Overall: {page_acc:.2f}% | (Year Acc: {y_acc:.0f}%,"
        f" Sem Acc: {s_acc:.0f}%)"
    )

  # 3. Category Summary
  print("\n" + "-" * 65)
  print("🏷️ 3. CATEGORY-LEVEL ACCURACY BREAKDOWN")
  print("-" * 65)
  for cat, fields in category_stats.items():
    total_fields = sum(s["total"] for s in fields.values())
    correct_fields = sum(s["correct"] for s in fields.values())
    cat_acc = (
        (correct_fields / total_fields * 100) if total_fields > 0 else 0
    )

    name_th_acc = (
        (fields["name_th"]["correct"] / fields["name_th"]["total"] * 100)
        if fields["name_th"]["total"] > 0
        else 0
    )
    name_en_acc = (
        (fields["name_en"]["correct"] / fields["name_en"]["total"] * 100)
        if fields["name_en"]["total"] > 0
        else 0
    )

    print(f"► Category: {cat}")
    print(
        f"  └─ Overall: {cat_acc:.2f}% | Name TH Acc: {name_th_acc:.1f}% | Name"
        f" EN Acc: {name_en_acc:.1f}%"
    )

  print("\n" + "=" * 65)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Evaluate Extracted JSON vs Ground Truth JSON"
  )
  parser.add_argument(
      "prediction", type=str, help="Path to extracted/consolidated JSON file"
  )
  parser.add_argument(
      "--gt",
      "-g",
      type=str,
      default="ground_truth",
      help=(
          "Path to Ground Truth directory or specific JSON file (Default:"
          " 'ground_truth')"
      ),
  )

  args = parser.parse_args()

  run_evaluation(Path(args.prediction), Path(args.gt))