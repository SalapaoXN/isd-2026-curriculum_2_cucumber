import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_system.evaluation import evaluate_extracted_from_files


def test_evaluate_extracted_from_files_computes_course_metrics(tmp_path):
    ground_truth_path = tmp_path / "ground_truth.json"
    prediction_path = tmp_path / "prediction.json"

    ground_truth = {
        "source": "gt",
        "program": "DSBA",
        "plan": "no_coop",
        "courses": [
            {
                "code": "CS101",
                "name_th": "แคลคูลัส",
                "name_en": "CALCULUS",
                "credits": "3(3-0-6)",
                "year": 1,
                "semester": 1,
            }
        ],
    }
    prediction = {
        "source": "ocr",
        "program": "DSBA",
        "plan": "no_coop",
        "courses": [
            {
                "code": "CS101",
                "name_th": "แคลคูลัส",
                "name_en": "CALCULUS",
                "credits": "3(3-0-6)",
                "year": 1,
                "semester": 1,
            }
        ],
    }

    ground_truth_path.write_text(json.dumps(ground_truth), encoding="utf-8")
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")

    result = evaluate_extracted_from_files(ground_truth_path, prediction_path)

    assert result["file"] == prediction_path.name
    assert result["matched_courses"] == 1
    assert result["reference_courses"] == 1
    assert result["prediction_courses"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["exact_match"] is True
