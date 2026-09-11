import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import llm_spell_corrector


def make_record(course_code="06000001"):
    return {
        "course_code": course_code,
        "name_th": "ชื่อวิชา",
        "name_en": "Original name",
        "desc_th": "คำอธิบายเดิม",
        "desc_en": "Original description",
        "note": "หมายเหตุเดิม",
        "credits": 3,
        "year": 1,
        "semester": 1,
        "flexible_year_semester_raw": None,
        "category": "วิชาแกน",
        "type": "required",
        "prerequisite": {"course_code": "06000000"},
        "program": "IT",
        "plan_key": "no_coop",
        "provenance": [{"source": "curriculum.pdf", "page": 4}],
    }


class FakeModels:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_content(self, *, model, contents):
        self.calls.append({"model": model, "contents": contents})
        return SimpleNamespace(text=next(self.responses))


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


class LlmSpellCorrectorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_document(self, records):
        path = self.directory / "curriculum.json"
        document = {"metadata": {"source": "test"}, "courses": records}
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        return path, document

    def response_for(self, records):
        return json.dumps(records, ensure_ascii=False)

    def run_corrector(self, path, responses, output_dir=None):
        client = FakeClient(responses)
        with patch("llm_spell_corrector.genai.Client", return_value=client):
            output_path = llm_spell_corrector.correct_json_file(path, output_dir=output_dir)
        return output_path, client

    def assert_no_outputs(self):
        self.assertFalse((self.directory / "curriculum_corrected.json").exists())
        self.assertFalse((self.directory / "curriculum_corrections.json").exists())

    def test_allowed_name_en_correction_succeeds_and_is_logged(self):
        records = [make_record()]
        path, original = self.write_document(records)
        corrected = copy.deepcopy(records)
        corrected[0]["name_en"] = "Corrected name"

        output_path, client = self.run_corrector(path, [self.response_for(corrected)])

        self.assertEqual(client.models.calls[0]["model"], "gemini-3.5-flash-lite")
        self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["courses"], corrected)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), original)
        self.assertEqual(
            json.loads((self.directory / "curriculum_corrections.json").read_text(encoding="utf-8")),
            [{"course_code": "06000001", "field": "name_en", "before": "Original name", "after": "Corrected name"}],
        )

    def test_allowed_desc_th_correction_succeeds(self):
        records = [make_record()]
        path, _ = self.write_document(records)
        corrected = copy.deepcopy(records)
        corrected[0]["desc_th"] = "คำอธิบายที่แก้ไขแล้ว"

        output_path, _ = self.run_corrector(path, [self.response_for(corrected)])

        result = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(result["courses"][0]["desc_th"], "คำอธิบายที่แก้ไขแล้ว")
        log = json.loads((self.directory / "curriculum_corrections.json").read_text(encoding="utf-8"))
        self.assertEqual(log[0]["field"], "desc_th")

    def test_output_dir_writes_both_artifacts(self):
        records = [make_record()]
        path, _ = self.write_document(records)
        corrected = copy.deepcopy(records)
        corrected[0]["name_en"] = "Corrected name"
        output_dir = self.directory / "output"
        output_dir.mkdir()

        output_path, _ = self.run_corrector(
            path,
            [self.response_for(corrected)],
            output_dir=output_dir,
        )

        self.assertEqual(output_path, output_dir / "curriculum_corrected.json")
        self.assertTrue((output_dir / "curriculum_corrected.json").is_file())
        self.assertTrue((output_dir / "curriculum_corrections.json").is_file())
        self.assertFalse((self.directory / "curriculum_corrections.json").exists())

    def test_output_dir_does_not_modify_input(self):
        records = [make_record()]
        path, _ = self.write_document(records)
        original_bytes = path.read_bytes()
        output_dir = self.directory / "output"

        self.run_corrector(path, [self.response_for(records)], output_dir=output_dir)

        self.assertEqual(path.read_bytes(), original_bytes)

    def test_output_dir_is_created_when_missing(self):
        records = [make_record()]
        path, _ = self.write_document(records)
        output_dir = self.directory / "nested" / "safe" / "output"

        self.run_corrector(path, [self.response_for(records)], output_dir=output_dir)

        self.assertTrue(output_dir.is_dir())
        self.assertTrue((output_dir / "curriculum_corrected.json").is_file())
        self.assertTrue((output_dir / "curriculum_corrections.json").is_file())

    def test_course_code_mutation_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document(records)
        changed = copy.deepcopy(records)
        changed[0]["course_code"] = "06999999"

        with self.assertRaises(ValueError):
            self.run_corrector(path, [self.response_for(changed)])
        self.assert_no_outputs()

    def test_credits_mutation_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document(records)
        changed = copy.deepcopy(records)
        changed[0]["credits"] = 4

        with self.assertRaises(ValueError):
            self.run_corrector(path, [self.response_for(changed)])
        self.assert_no_outputs()

    def test_prerequisite_mutation_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document(records)
        changed = copy.deepcopy(records)
        changed[0]["prerequisite"]["course_code"] = "06000002"

        with self.assertRaises(ValueError):
            self.run_corrector(path, [self.response_for(changed)])
        self.assert_no_outputs()

    def test_provenance_mutation_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document(records)
        changed = copy.deepcopy(records)
        changed[0]["provenance"][0]["page"] = 99

        with self.assertRaises(ValueError):
            self.run_corrector(path, [self.response_for(changed)])
        self.assert_no_outputs()

    def test_dropped_record_is_rejected(self):
        records = [make_record("06000001"), make_record("06000002")]
        path, _ = self.write_document(records)

        with self.assertRaises(ValueError):
            self.run_corrector(path, [self.response_for(records[:1])])
        self.assert_no_outputs()

    def test_added_record_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document(records)
        added = records + [make_record("06000002")]

        with self.assertRaises(ValueError):
            self.run_corrector(path, [self.response_for(added)])
        self.assert_no_outputs()

    def test_invalid_json_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document(records)

        with self.assertRaises(ValueError):
            self.run_corrector(path, ["not valid json"])
        self.assert_no_outputs()

    def test_unchanged_input_has_empty_correction_log(self):
        records = [make_record()]
        path, original = self.write_document(records)

        output_path, _ = self.run_corrector(path, [self.response_for(records)])

        self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), original)
        self.assertEqual(
            json.loads((self.directory / "curriculum_corrections.json").read_text(encoding="utf-8")),
            [],
        )

    def test_records_are_sent_in_bounded_batches_in_original_order(self):
        records = [make_record(f"060000{i:03d}") for i in range(11)]
        path, _ = self.write_document(records)

        _, client = self.run_corrector(
            path,
            [self.response_for(records[:10]), self.response_for(records[10:])],
        )

        self.assertEqual(len(client.models.calls), 2)
        self.assertEqual(
            [len(json.loads(call["contents"][1])) for call in client.models.calls],
            [10, 1],
        )


if __name__ == "__main__":
    unittest.main()
