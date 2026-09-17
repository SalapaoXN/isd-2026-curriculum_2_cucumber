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
    @staticmethod
    def validated_text(original, corrected, field="name_en"):
        before = [{"unit_index": 0, "field": field, "text": original}]
        after = [{"unit_index": 0, "field": field, "text": corrected}]
        return llm_spell_corrector._validate_batch(before, after, 1, 0)[0]["text"]

    def test_terminal_suffix_deletion_is_rejected(self):
        self.assertEqual(
            self.validated_text("COURSE NAME 3", "COURSE NAME"),
            "COURSE NAME 3",
        )

    def test_terminal_numeric_suffix_accepts_only_one_digit(self):
        cases = {
            "1": "1",
            "9": "9",
            "12": None,
            "23": None,
            "COURSE 1": "1",
            "COURSE 12": None,
            "COURSE1": None,
            "": None,
            None: None,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(
                    llm_spell_corrector._terminal_numeric_suffix(value), expected
                )

    def test_terminal_suffix_addition_is_rejected(self):
        self.assertEqual(
            self.validated_text("COURSE NAME", "COURSE NAME 3"),
            "COURSE NAME",
        )

    def test_terminal_suffix_substitution_is_rejected(self):
        self.assertEqual(
            self.validated_text("COURSE NAME 3", "COURSE NAME 4"),
            "COURSE NAME 3",
        )

    def test_same_terminal_suffix_and_suffix_free_corrections_are_accepted(self):
        self.assertEqual(
            self.validated_text("COURSE NANE 3", "COURSE NAME 3"),
            "COURSE NAME 3",
        )
        self.assertEqual(
            self.validated_text("COURSE NANE", "COURSE NAME"),
            "COURSE NAME",
        )
        self.assertEqual(
            self.validated_text("COURSE 12", "COURSE"),
            "COURSE",
        )

    def test_terminal_suffix_guard_applies_to_thai_names(self):
        self.assertEqual(
            self.validated_text(
                "โครงงานปัญญาประดิษฐ์ 1",
                "โครงงานปัญญาประดิษฐ์",
                field="name_th",
            ),
            "โครงงานปัญญาประดิษฐ์ 1",
        )

    def test_mixed_batch_preserves_unsafe_unit_and_applies_valid_unit(self):
        before = [
            {"unit_index": 0, "field": "name_th", "text": "ชื่อวิชา 1"},
            {"unit_index": 1, "field": "name_en", "text": "COURSE NANE"},
        ]
        after = [
            {"unit_index": 0, "field": "name_th", "text": "ชื่อวิชา"},
            {"unit_index": 1, "field": "name_en", "text": "COURSE NAME"},
        ]
        validated = llm_spell_corrector._validate_batch(before, after, 1, 0)
        self.assertEqual(validated[0]["text"], "ชื่อวิชา 1")
        self.assertEqual(validated[1]["text"], "COURSE NAME")

        corrected, corrections = llm_spell_corrector._reconstruct_document(
            {"courses": [make_record(name_th="ชื่อวิชา 1", name_en="COURSE NANE")]},
            {
                (original["field"], original["text"]): unit["text"]
                for original, unit in zip(before, validated.values())
            },
        )
        self.assertEqual(corrected["courses"][0]["name_th"], "ชื่อวิชา 1")
        self.assertEqual(corrected["courses"][0]["name_en"], "COURSE NAME")
        self.assertEqual([entry["field"] for entry in corrections], ["name_en"])

    def test_canonical_corrections_apply_deterministically_in_reconstruction(self):
        for (program, course_code, field, before), after in (
            llm_spell_corrector.CANONICAL_NAME_CORRECTIONS.items()
        ):
            with self.subTest(program=program, course_code=course_code, field=field):
                record = make_record(course_code=course_code)
                record.pop("program")
                record[field] = before
                corrected, applied = llm_spell_corrector._reconstruct_document(
                    {"program": program, "courses": [record]},
                    {(field, before): "LLM candidate"},
                )
                self.assertEqual(corrected["courses"][0][field], after)
                self.assertEqual(applied[0]["after"], after)

    def test_canonical_corrections_apply_deterministically_in_replay(self):
        for (program, course_code, field, before), after in (
            llm_spell_corrector.CANONICAL_NAME_CORRECTIONS.items()
        ):
            with self.subTest(program=program, course_code=course_code, field=field):
                record = make_record(course_code=course_code)
                record.pop("program")
                record[field] = before
                corrected, applied = llm_spell_corrector.apply_corrections(
                    {"program": program, "courses": [record]},
                    [{
                        "course_code": course_code,
                        "field": field,
                        "before": before,
                        "after": "LLM candidate",
                    }],
                )
                self.assertEqual(corrected["courses"][0][field], after)
                self.assertEqual(applied[0]["after"], after)

    def test_phase_6c2_pins_apply_exact_verified_values(self):
        expected_pins = {
            ("IT", "06016412", "name_en", "COMPUTER ORCANIZATON AND OPERATING SSTEM"):
                "COMPUTER ORGANIZATION AND OPERATING SYSTEM",
            ("IT", "06016466", "name_en", "NETWORK AND SYSTEM TROUBLE SHOOTNG"):
                "NETWORK AND SYSTEM TROUBLE SHOOTING",
            ("AIT", "06046413", "name_th", "ปัญญา ประดิษฐ์และอินเทอร์เน็ตประสานสรรพสิง"):
                "ปัญญาประดิษฐ์และอินเทอร์เน็ตประสานสรรพสิ่ง",
            (
                "DSBA",
                "06026260",
                "name_en",
                "OVERSEA COOPERATIVE EDUCATION IN DATA SCIENCE AND BUSIESS ANALYTICS",
            ): "OVERSEA COOPERATIVE EDUCATION IN DATA SCIENCE AND BUSINESS ANALYTICS",
            ("GENED", "90642056", "name_en", "ST EPLDEMICS IN THE 21 CENTURV"):
                "EPIDEMICS IN THE 21ST CENTURY",
            ("GENED", "90642045", "name_en", "BE MV BEV."): "BEVERAGE",
            ("IT", "06016418", "name_th", "การพัฒนาเว็บฝังเซิร์ฟเวอร์"):
                "การพัฒนาเว็บฝั่งเซิร์ฟเวอร์",
            (
                "IT",
                "06016442",
                "name_th",
                "การออกแบบฮาร์ดแวร์สำหรับอินเทอร์เน็ตแห่งสรรพสิง",
            ): "การออกแบบฮาร์ดแวร์สำหรับอินเทอร์เน็ตแห่งสรรพสิ่ง",
            (
                "IT",
                "06016443",
                "name_th",
                "การวิเคราะห์ข้อมูลและแอปพลิเคชันสำหรับอินเทอร์เน็ตแห่งสรรพสิง",
            ): "การวิเคราะห์ข้อมูลและแอปพลิเคชันสำหรับอินเทอร์เน็ตแห่งสรรพสิ่ง",
            ("IT", "90643021", "name_th", "ผู้ ประกอบการสมัยใหม่"):
                "ผู้ประกอบการสมัยใหม่",
            ("GENED", "90642134", "name_en", "KING MONGKUTS REIGN STUDV"):
                "KING MONGKUTS REIGN STUDY",
        }
        for (program, course_code, field, before), after in expected_pins.items():
            with self.subTest(program=program, course_code=course_code, field=field):
                record = make_record(course_code=course_code)
                record.pop("program")
                record[field] = before
                document = {"program": program, "courses": [record]}
                reconstructed, reconstruction_log = (
                    llm_spell_corrector._reconstruct_document(
                        document,
                        {(field, before): "untrusted candidate"},
                    )
                )
                self.assertEqual(reconstructed["courses"][0][field], after)
                self.assertEqual(reconstruction_log[0]["after"], after)

                replayed, replay_log = llm_spell_corrector.apply_corrections(
                    document,
                    [{
                        "course_code": course_code,
                        "field": field,
                        "before": before,
                        "after": "untrusted candidate",
                    }],
                )
                self.assertEqual(replayed["courses"][0][field], after)
                self.assertEqual(replay_log[0]["after"], after)

    def test_canonical_rules_fail_closed_on_identity_and_before(self):
        for program, course_code, field, before in (
            key for key in llm_spell_corrector.CANONICAL_NAME_CORRECTIONS
        ):
            wrong_field = "name_en" if field == "name_th" else "name_th"
            cases = (
                ("WRONG", course_code, field, before),
                (program, "99999999", field, before),
                (program, course_code, wrong_field, before),
                (program, course_code, field, f"{before} changed"),
            )
            for wrong_program, wrong_code, wrong_name_field, current in cases:
                with self.subTest(
                    program=program,
                    course_code=course_code,
                    field=wrong_name_field,
                    current=current,
                ):
                    record = make_record(course_code=wrong_code)
                    record.pop("program")
                    record[wrong_name_field] = current
                    corrected, applied = llm_spell_corrector._reconstruct_document(
                        {"program": wrong_program, "courses": [record]},
                        {(wrong_name_field, current): "LLM candidate"},
                    )
                    self.assertEqual(
                        corrected["courses"][0][wrong_name_field], "LLM candidate"
                    )
                    self.assertEqual(len(applied), 1)

    def test_reconstruct_still_applies_unregistered_correction(self):
        record = make_record(name_en="Original name")
        corrected, applied = llm_spell_corrector._reconstruct_document(
            {"program": "IT", "courses": [record]},
            {("name_en", "Original name"): "Corrected name"},
        )
        self.assertEqual(corrected["courses"][0]["name_en"], "Corrected name")
        self.assertEqual(
            applied,
            [{
                "course_code": "06000001",
                "field": "name_en",
                "before": "Original name",
                "after": "Corrected name",
            }],
        )

    def test_placeholder_names_are_not_corrected_or_logged(self):
        for field, placeholder in (("name_th", "ไม่ระบุ"), ("name_en", "N/A")):
            with self.subTest(field=field):
                record = make_record(course_code="06000001")
                record[field] = placeholder
                corrected, reconstructed = llm_spell_corrector._reconstruct_document(
                    {"program": "IT", "courses": [record]},
                    {(field, placeholder): "Hallucinated course name"},
                )
                self.assertEqual(corrected["courses"][0][field], placeholder)
                self.assertEqual(reconstructed, [])

                replayed, applied = llm_spell_corrector.apply_corrections(
                    {"courses": [record]},
                    [{
                        "course_code": "06000001",
                        "field": field,
                        "before": placeholder,
                        "after": "Hallucinated course name",
                    }],
                )
                self.assertEqual(replayed["courses"][0][field], placeholder)
                self.assertEqual(applied, [])

    def test_literal_preserve_values_reject_changed_candidates(self):
        for (program, course_code, field, source_value), preserved_value in (
            llm_spell_corrector.LITERAL_PRESERVE_VALUES.items()
        ):
            with self.subTest(program=program, course_code=course_code, field=field):
                record = make_record(course_code=course_code)
                record.pop("program")
                record[field] = source_value
                reconstructed, reconstruction_log = llm_spell_corrector._reconstruct_document(
                    {"program": program, "courses": [record]},
                    {(field, source_value): "LLM candidate"},
                )
                self.assertEqual(
                    reconstructed["courses"][0][field], preserved_value
                )
                self.assertEqual(reconstruction_log, [])
                corrected, applied = llm_spell_corrector.apply_corrections(
                    {"program": program, "courses": [record]},
                    [{
                        "course_code": course_code,
                        "field": field,
                        "before": source_value,
                        "after": f"{source_value} changed",
                    }],
                )
                self.assertEqual(corrected["courses"][0][field], preserved_value)
                self.assertEqual(applied, [])

    def test_phase_6c2_preserves_gened_source_titles(self):
        for course_code, source_value, candidate in (
            ("90642126", "SURVIVORS", "SURVIVAL"),
            ("90642154", "FALL ABLE", "FALLABLE"),
        ):
            with self.subTest(course_code=course_code):
                record = make_record(course_code=course_code, name_en=source_value)
                record.pop("program")
                document = {"program": "GENED", "courses": [record]}

                reconstructed, reconstruction_log = (
                    llm_spell_corrector._reconstruct_document(
                        document,
                        {("name_en", source_value): candidate},
                    )
                )
                self.assertEqual(
                    reconstructed["courses"][0]["name_en"], source_value
                )
                self.assertEqual(reconstruction_log, [])

                replayed, replay_log = llm_spell_corrector.apply_corrections(
                    document,
                    [{
                        "course_code": course_code,
                        "field": "name_en",
                        "before": source_value,
                        "after": candidate,
                    }],
                )
                self.assertEqual(replayed["courses"][0]["name_en"], source_value)
                self.assertEqual(replay_log, [])

    def test_literal_preserve_rules_fail_closed_on_identity_and_before(self):
        for program, course_code, field, before in (
            key for key in llm_spell_corrector.LITERAL_PRESERVE_VALUES
        ):
            wrong_field = "name_en" if field == "name_th" else "name_th"
            cases = (
                ("WRONG", course_code, field, before),
                (program, "99999999", field, before),
                (program, course_code, wrong_field, before),
                (program, course_code, field, f"{before} changed"),
            )
            for wrong_program, wrong_code, wrong_name_field, current in cases:
                with self.subTest(
                    program=program,
                    course_code=course_code,
                    field=wrong_name_field,
                    current=current,
                ):
                    record = make_record(course_code=wrong_code)
                    record.pop("program")
                    record[wrong_name_field] = current
                    corrected, applied = llm_spell_corrector._reconstruct_document(
                        {"program": wrong_program, "courses": [record]},
                        {(wrong_name_field, current): "LLM candidate"},
                    )
                    self.assertEqual(
                        corrected["courses"][0][wrong_name_field], "LLM candidate"
                    )
                    self.assertEqual(len(applied), 1)

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
        if output_dir is None:
            output_dir = self.directory
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

    def test_default_output_paths_use_canonical_llm_layer_for_all_nested_scopes(self):
        cases = (
            "outputs/consolidated/ait/full/merged_ait_no_plan_full.json",
            "outputs/consolidated/bit/coop/full/merged_bit_coop_full.json",
            "outputs/consolidated/dsba/no_coop/full/merged_dsba_no_coop_full.json",
        )
        for source in cases:
            with self.subTest(source=source):
                corrected, corrections = llm_spell_corrector._output_paths(Path(source), None)
                self.assertEqual(corrected.parent, llm_spell_corrector.LLM_OUTPUT_DIR)
                self.assertEqual(corrections.parent, llm_spell_corrector.LLM_OUTPUT_DIR)
                self.assertEqual(corrected.name, f"{Path(source).stem}_corrected.json")
                self.assertEqual(corrections.name, f"{Path(source).stem}_corrections.json")
                self.assertNotEqual(corrected.parent, Path(source).parent)

    def test_explicit_output_directory_still_overrides_canonical_default(self):
        source = Path("outputs/consolidated/bit/coop/full/merged_bit_coop_full.json")
        corrected, corrections = llm_spell_corrector._output_paths(source, self.directory)
        self.assertEqual(corrected.parent, self.directory)
        self.assertEqual(corrections.parent, self.directory)

    def test_default_routing_does_not_write_beside_nested_input(self):
        source = self.directory / "outputs" / "consolidated" / "ait" / "full" / "merged_ait_no_plan_full.json"
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps({"courses": [make_record()]}), encoding="utf-8")

        corrected, corrections = llm_spell_corrector._output_paths(source, None)

        self.assertEqual(corrected.parent, llm_spell_corrector.LLM_OUTPUT_DIR)
        self.assertEqual(corrections.parent, llm_spell_corrector.LLM_OUTPUT_DIR)
        self.assertFalse((source.parent / corrected.name).exists())
        self.assertFalse((source.parent / corrections.name).exists())

    def test_discovery_finds_only_full_consolidated_files_in_deterministic_order(self):
        consolidated = self.directory / "outputs" / "consolidated"
        first = consolidated / "it" / "coop" / "full" / "merged_it_coop_full.json"
        second = consolidated / "ait" / "full" / "merged_ait_no_plan_full.json"
        page_range = consolidated / "it" / "coop" / "page_ranges" / "merged_it_coop_full.json"
        correction_log = consolidated / "it" / "coop" / "full" / "merged_it_coop_corrections.json"
        for path in (first, second, page_range, correction_log):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"courses": []}', encoding="utf-8")

        self.assertEqual(
            llm_spell_corrector.discover_consolidated_inputs(consolidated),
            [second, first],
        )

    def test_discovery_supports_partial_corpora(self):
        consolidated = self.directory / "outputs" / "consolidated"
        path = consolidated / "it" / "coop" / "full" / "merged_it_coop_full.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"courses": []}', encoding="utf-8")

        self.assertEqual(llm_spell_corrector.discover_consolidated_inputs(consolidated), [path])

    def test_discovery_fails_clearly_when_no_full_files_exist(self):
        with self.assertRaisesRegex(
            FileNotFoundError, "No full consolidated curriculum files"
        ):
            llm_spell_corrector.discover_consolidated_inputs(self.directory / "missing")

    def test_discovery_rejects_invalid_full_json(self):
        path = self.directory / "outputs" / "consolidated" / "it" / "full" / "merged_it_full.json"
        path.parent.mkdir(parents=True)
        path.write_text("not json", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            llm_spell_corrector.discover_consolidated_inputs(path.parents[3])

    def test_zero_argument_cli_discovers_and_defaults_to_llm_output(self):
        discovered = [Path("outputs/consolidated/it/coop/full/merged_it_coop_full.json")]
        with patch.object(llm_spell_corrector, "discover_consolidated_inputs", return_value=discovered) as discover, patch.object(
            llm_spell_corrector, "correct_json_files", return_value=[]
        ) as correct:
            self.assertEqual(llm_spell_corrector.main([]), 0)

        discover.assert_called_once_with()
        correct.assert_called_once_with(discovered, output_dir=llm_spell_corrector.LLM_OUTPUT_DIR)

    def test_zero_argument_cli_fails_nonzero_when_discovery_fails(self):
        with patch.object(
            llm_spell_corrector,
            "discover_consolidated_inputs",
            side_effect=FileNotFoundError("No full consolidated curriculum files"),
        ):
            self.assertEqual(llm_spell_corrector.main([]), 1)

    def test_explicit_inputs_and_output_override_remain_supported(self):
        explicit = [self.directory / "input.json"]
        override = self.directory / "custom-output"
        with patch.object(
            llm_spell_corrector, "correct_json_files", return_value=[]
        ) as correct:
            llm_spell_corrector.main([str(explicit[0]), "--output-dir", str(override)])

        correct.assert_called_once_with(
            [str(explicit[0])], output_dir=override
        )

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

    def test_empty_corrected_name_th_is_rejected_before_output(self):
        path, _ = self.write_document("curriculum.json", [make_record()])

        def empty_name_th(contents):
            payload = json.loads(contents[1])
            payload[0]["text"] = ""
            return self.response_for_payload(payload)

        with self.assertRaisesRegex(ValueError, "empty text"):
            self.run_corrector(path, [empty_name_th])
        self.assert_no_outputs()

    def test_whitespace_corrected_name_en_is_rejected_before_output(self):
        path, _ = self.write_document("curriculum.json", [make_record()])

        def whitespace_name_en(contents):
            payload = json.loads(contents[1])
            for unit in payload:
                if unit["field"] == "name_en":
                    unit["text"] = " \t\n"
            return self.response_for_payload(payload)

        with self.assertRaisesRegex(ValueError, "empty text"):
            self.run_corrector(path, [whitespace_name_en])
        self.assert_no_outputs()

    def test_empty_original_text_does_not_require_a_replacement(self):
        path, _ = self.write_document(
            "curriculum.json",
            [make_record(name_th="", name_en="", desc_th="", desc_en="")],
        )

        output_path, client = self.run_corrector(path, [])

        result = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(result["courses"][0]["name_th"], "")
        self.assertEqual(result["courses"][0]["name_en"], "")
        self.assertEqual(client.models.calls, [])

    def test_whitespace_only_original_text_does_not_require_a_replacement(self):
        before = [{"unit_index": 0, "field": "name_en", "text": "   "}]
        after = [{"unit_index": 0, "field": "name_en", "text": ""}]

        validated = llm_spell_corrector._validate_batch(before, after, 1, 0)

        self.assertEqual(validated[0]["text"], "")

    def test_reviewed_corrections_reject_empty_text_for_all_supported_text_fields(self):
        for field in ("name_th", "name_en", "desc_th", "desc_en"):
            record = make_record()
            before = record[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "empty text"):
                llm_spell_corrector.apply_corrections(
                    {"courses": [record]},
                    [
                        {
                            "course_code": record["course_code"],
                            "field": field,
                            "before": before,
                            "after": "",
                        }
                    ],
                )

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
