import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import llm_spell_corrector


def make_record(
    course_code="06000001",
    name_th="ชื่อวิชา",
    name_en="Original name",
    desc_th="คำอธิบายเดิม",
    desc_en="Original description",
    note="หมายเหตุเดิม",
):
    return {
        "course_code": course_code,
        "name_th": name_th,
        "name_en": name_en,
        "desc_th": desc_th,
        "desc_en": desc_en,
        "note": note,
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
        "source_provenance": [{"source": "curriculum.pdf", "page": 4}],
    }


class FakeModels:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.response_index = 0

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if callable(self.responses):
            response_text = self.responses(contents)
        else:
            response_spec = self.responses[self.response_index]
            self.response_index += 1
            response_text = response_spec(contents) if callable(response_spec) else response_spec
        return SimpleNamespace(text=response_text)


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


class LlmSpellCorrectorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_document(self, filename, records):
        path = self.directory / filename
        document = {"metadata": {"source": "test"}, "courses": records}
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        return path, document

    def payload_for(self, records, start_index=0):
        units = []
        for record in records:
            for field in llm_spell_corrector.TEXT_FIELDS_ORDER:
                text = record.get(field)
                if not isinstance(text, str) or text == "":
                    continue
                units.append(
                    {
                        "unit_index": start_index + len(units),
                        "field": field,
                        "text": text,
                    }
                )
        return units

    def response_for_payload(self, payload):
        return json.dumps(payload, ensure_ascii=False)

    def response_for_records(self, records, start_index=0):
        return self.response_for_payload(self.payload_for(records, start_index))

    def replacement_response(self, field, before, after):
        def responder(contents):
            payload = json.loads(contents[1])
            for unit in payload:
                if unit["field"] == field and unit["text"] == before:
                    unit["text"] = after
            return self.response_for_payload(payload)

        return responder

    def run_files(self, paths, responses, output_dir=None):
        client = FakeClient(responses)
        with patch("llm_spell_corrector.genai.Client", return_value=client):
            output_paths = llm_spell_corrector.correct_json_files(
                paths,
                output_dir=output_dir,
            )
        return output_paths, client

    def run_corrector(self, path, responses, output_dir=None):
        output_paths, client = self.run_files([path], responses, output_dir)
        return output_paths[0], client

    def assert_no_outputs(self, directory=None):
        directory = self.directory if directory is None else directory
        self.assertFalse(list(directory.glob("*_corrected.json")))
        self.assertFalse(list(directory.glob("*_corrections.json")))

    def test_reviewed_corrections_apply_all_supported_text_fields_and_are_idempotent(self):
        original = {"metadata": {"program": "IT", "plan": "coop"}, "courses": [make_record()]}
        corrections = [
            {
                "course_code": "06000001",
                "field": "name_th",
                "before": "ชื่อวิชา",
                "after": "ชื่อวิชาที่แก้แล้ว",
            },
            {
                "course_code": "06000001",
                "field": "name_en",
                "before": "Original name",
                "after": "Corrected name",
            },
            {
                "course_code": "06000001",
                "field": "desc_th",
                "before": "คำอธิบายเดิม",
                "after": "คำอธิบายที่แก้แล้ว",
            },
            {
                "course_code": "06000001",
                "field": "desc_en",
                "before": "Original description",
                "after": "Corrected description",
            },
        ]

        corrected, applied = llm_spell_corrector.apply_corrections(
            original,
            corrections,
        )

        self.assertEqual(len(applied), 4)
        self.assertEqual(corrected["courses"][0]["name_en"], "Corrected name")
        self.assertEqual(corrected["courses"][0]["desc_th"], "คำอธิบายที่แก้แล้ว")
        self.assertEqual(corrected["courses"][0]["desc_en"], "Corrected description")
        for field in (
            "course_code",
            "credits",
            "year",
            "semester",
            "prerequisite",
            "provenance",
            "source_provenance",
        ):
            self.assertEqual(corrected["courses"][0][field], original["courses"][0][field])

        reapplied, second_applied = llm_spell_corrector.apply_corrections(
            corrected,
            corrections,
        )
        self.assertEqual(reapplied, corrected)
        self.assertEqual(second_applied, [])

    def test_reviewed_correction_with_unknown_field_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "unsupported field"):
            llm_spell_corrector.apply_corrections(
                {"courses": [make_record()]},
                [
                    {
                        "course_code": "06000001",
                        "field": "credits",
                        "before": "3",
                        "after": "4",
                    }
                ],
            )

    def test_normal_text_correction_succeeds_and_is_logged(self):
        records = [make_record()]
        path, original = self.write_document("curriculum.json", records)

        output_path, client = self.run_corrector(
            path,
            [self.replacement_response("name_en", "Original name", "Corrected name")],
        )

        result = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(result["courses"][0]["name_en"], "Corrected name")
        self.assertEqual(result["courses"][0]["course_code"], original["courses"][0]["course_code"])
        self.assertEqual(result["courses"][0]["credits"], original["courses"][0]["credits"])
        self.assertEqual(result["courses"][0]["prerequisite"], original["courses"][0]["prerequisite"])
        self.assertEqual(result["courses"][0]["provenance"], original["courses"][0]["provenance"])
        self.assertEqual(result["courses"][0]["source_provenance"], original["courses"][0]["source_provenance"])
        self.assertEqual(result["courses"][0]["desc_th"], original["courses"][0]["desc_th"])
        self.assertEqual(result["courses"][0]["desc_en"], original["courses"][0]["desc_en"])
        self.assertEqual(result["courses"][0]["note"], original["courses"][0]["note"])
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), original)
        self.assertEqual(
            json.loads((self.directory / "curriculum_corrections.json").read_text(encoding="utf-8")),
            [{"course_code": "06000001", "field": "name_en", "before": "Original name", "after": "Corrected name"}],
        )
        self.assertEqual(client.models.calls[0]["model"], "gemini-3.5-flash-lite")
        self.assertEqual(client.models.calls[0]["config"], {"temperature": 0})

    def test_identical_text_across_files_uses_one_unit_and_same_output(self):
        first_records = [make_record("06000001", name_th=None, name_en="Shared text", desc_th=None, desc_en=None, note=None)]
        second_records = [make_record("06000002", name_th=None, name_en="Shared text", desc_th=None, desc_en=None, note=None)]
        first_path, _ = self.write_document("first.json", first_records)
        second_path, _ = self.write_document("second.json", second_records)

        output_paths, client = self.run_files(
            [first_path, second_path],
            [self.replacement_response("name_en", "Shared text", "Shared corrected")],
        )

        self.assertEqual(len(client.models.calls), 1)
        sent_units = json.loads(client.models.calls[0]["contents"][1])
        self.assertEqual(sent_units, [{"unit_index": 0, "field": "name_en", "text": "Shared text"}])
        for output_path in output_paths:
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["courses"][0]["name_en"], "Shared corrected")
        first_log = json.loads((self.directory / "first_corrections.json").read_text(encoding="utf-8"))
        second_log = json.loads((self.directory / "second_corrections.json").read_text(encoding="utf-8"))
        self.assertEqual(first_log[0]["after"], second_log[0]["after"])

    def test_different_original_text_remains_separate(self):
        first_path, _ = self.write_document(
            "first.json",
            [make_record("06000001", name_th=None, name_en="Text A", desc_th=None, desc_en=None, note=None)],
        )
        second_path, _ = self.write_document(
            "second.json",
            [make_record("06000002", name_th=None, name_en="Text B", desc_th=None, desc_en=None, note=None)],
        )

        _, client = self.run_files(
            [first_path, second_path],
            [
                self.replacement_response("name_en", "Text A", "Text A corrected")
            ],
        )

        sent_units = json.loads(client.models.calls[0]["contents"][1])
        self.assertEqual({unit["text"] for unit in sent_units}, {"Text A", "Text B"})
        self.assertEqual(len(sent_units), 2)

    def test_unchanged_text_creates_no_correction_entry(self):
        records = [make_record()]
        path, _ = self.write_document("curriculum.json", records)

        self.run_corrector(path, [lambda contents: contents[1]])

        self.assertEqual(
            json.loads((self.directory / "curriculum_corrections.json").read_text(encoding="utf-8")),
            [],
        )

    def test_gemini_payload_contains_no_immutable_fields(self):
        records = [make_record()]
        path, _ = self.write_document("curriculum.json", records)

        _, client = self.run_corrector(path, [lambda contents: contents[1]])

        payload_text = client.models.calls[0]["contents"][1]
        self.assertNotIn("source_provenance", payload_text)
        self.assertNotIn("provenance", payload_text)
        self.assertNotIn("credits", payload_text)
        self.assertNotIn("prerequisite", payload_text)
        self.assertNotIn("course_code", payload_text)
        self.assertNotIn("desc_th", payload_text)
        self.assertNotIn("desc_en", payload_text)
        self.assertNotIn("note", payload_text)

    def test_descriptions_and_note_are_not_editable_or_logged(self):
        records = [make_record()]
        path, original = self.write_document("curriculum.json", records)
        observed_fields = []

        def correct_name_only(contents):
            payload = json.loads(contents[1])
            observed_fields.extend(unit["field"] for unit in payload)
            for unit in payload:
                if unit["field"] == "name_en":
                    unit["text"] = "Corrected name"
            return self.response_for_payload(payload)

        output_path, _ = self.run_corrector(path, [correct_name_only])

        result = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(set(observed_fields), {"name_th", "name_en"})
        for field in ("desc_th", "desc_en", "note"):
            self.assertEqual(result["courses"][0][field], original["courses"][0][field])
        log = json.loads((self.directory / "curriculum_corrections.json").read_text(encoding="utf-8"))
        self.assertEqual([entry["field"] for entry in log], ["name_en"])

    def test_course_code_mutation_attempt_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document("curriculum.json", records)

        def mutate(payload):
            payload[0]["course_code"] = "06999999"
            return payload

        with self.assertRaises(ValueError):
            self.run_corrector(
                path,
                [lambda contents: self.response_for_payload(mutate(json.loads(contents[1])))],
            )
        self.assert_no_outputs()

    def test_credits_mutation_attempt_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document("curriculum.json", records)

        with self.assertRaises(ValueError):
            self.run_corrector(
                path,
                [
                    lambda contents: self.response_for_payload(
                        [{**unit, "credits": 4} for unit in json.loads(contents[1])]
                    )
                ],
            )
        self.assert_no_outputs()

    def test_prerequisite_mutation_attempt_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document("curriculum.json", records)

        with self.assertRaises(ValueError):
            self.run_corrector(
                path,
                [
                    lambda contents: self.response_for_payload(
                        [{**unit, "prerequisite": "06000002"} for unit in json.loads(contents[1])]
                    )
                ],
            )
        self.assert_no_outputs()

    def test_provenance_mutation_attempt_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document("curriculum.json", records)

        with self.assertRaises(ValueError):
            self.run_corrector(
                path,
                [
                    lambda contents: self.response_for_payload(
                        [{**unit, "source_provenance": "changed"} for unit in json.loads(contents[1])]
                    )
                ],
            )
        self.assert_no_outputs()

    def test_missing_unit_index_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document("curriculum.json", records)

        def remove_index(payload):
            del payload[0]["unit_index"]
            return payload

        with self.assertRaises(ValueError):
            self.run_corrector(
                path,
                [lambda contents: self.response_for_payload(remove_index(json.loads(contents[1])))],
            )
        self.assert_no_outputs()

    def test_duplicate_unit_index_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document("curriculum.json", records)

        def duplicate_index(payload):
            payload[1]["unit_index"] = payload[0]["unit_index"]
            return payload

        with self.assertRaises(ValueError):
            self.run_corrector(
                path,
                [lambda contents: self.response_for_payload(duplicate_index(json.loads(contents[1])))],
            )
        self.assert_no_outputs()

    def test_unknown_unit_index_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document("curriculum.json", records)

        with self.assertRaises(ValueError):
            self.run_corrector(
                path,
                [lambda contents: self.response_for_payload(
                    [{**unit, "unit_index": 999} if offset == 0 else unit
                     for offset, unit in enumerate(json.loads(contents[1]))]
                )],
            )
        self.assert_no_outputs()

    def test_extra_non_editable_response_field_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document("curriculum.json", records)

        with self.assertRaises(ValueError):
            self.run_corrector(
                path,
                [lambda contents: self.response_for_payload(
                    [{**unit, "source_provenance": "not allowed"} for unit in json.loads(contents[1])]
                )],
            )
        self.assert_no_outputs()

    def test_dropped_unit_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document("curriculum.json", records)

        with self.assertRaises(ValueError):
            self.run_corrector(path, [lambda contents: self.response_for_payload(json.loads(contents[1])[:-1])])
        self.assert_no_outputs()

    def test_added_unit_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document("curriculum.json", records)

        with self.assertRaises(ValueError):
            self.run_corrector(path, [lambda contents: self.response_for_payload(
                json.loads(contents[1]) + [{"unit_index": 999, "field": "name_en", "text": "extra"}]
            )])
        self.assert_no_outputs()

    def test_invalid_json_is_rejected(self):
        records = [make_record()]
        path, _ = self.write_document("curriculum.json", records)

        with self.assertRaises(ValueError):
            self.run_corrector(path, ["not valid json"])
        self.assert_no_outputs()

    def test_multi_file_failure_writes_no_partial_artifacts(self):
        first_records = [
            make_record(f"060000{i:02d}", name_th=None, name_en=f"Text {i}", desc_th=None, desc_en=None, note=None)
            for i in range(26)
        ]
        second_records = [
            make_record(f"060001{i:02d}", name_th=None, name_en=f"Other {i}", desc_th=None, desc_en=None, note=None)
            for i in range(26)
        ]
        first_path, _ = self.write_document("first.json", first_records)
        second_path, _ = self.write_document("second.json", second_records)
        output_dir = self.directory / "generated"

        calls = 0

        def fail_on_second_batch(contents):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated API failure")
            return contents[1]

        with self.assertRaises(RuntimeError):
            self.run_files([first_path, second_path], fail_on_second_batch, output_dir=output_dir)
        self.assertFalse(output_dir.exists())

    def test_records_are_sent_in_bounded_unique_unit_batches(self):
        records = [
            make_record(f"060000{i:02d}", name_th=None, name_en=f"Text {i}", desc_th=None, desc_en=None, note=None)
            for i in range(51)
        ]
        path, _ = self.write_document("curriculum.json", records)

        progress = StringIO()
        with redirect_stdout(progress):
            _, client = self.run_corrector(
                path,
                [lambda contents: contents[1], lambda contents: contents[1]],
            )

        self.assertEqual(len(client.models.calls), 2)
        self.assertEqual(
            [len(json.loads(call["contents"][1])) for call in client.models.calls],
            [50, 1],
        )
        self.assertIn("[1/2] correcting 50 unique units...", progress.getvalue())
        self.assertIn("[1/2] done", progress.getvalue())
        self.assertIn("[2/2] correcting 1 unique units...", progress.getvalue())
        self.assertIn("[2/2] done", progress.getvalue())


if __name__ == "__main__":
    unittest.main()
