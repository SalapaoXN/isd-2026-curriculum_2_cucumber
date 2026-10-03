import unittest
import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main
from backend.hard_qa import HARD_INTERPRETATION_RESPONSE_JSON_SCHEMA, answer_hard_question
from backend.llm_sql_qa import ask_sql as run_ask_sql


DB_PATH = Path(__file__).parents[1] / "cucumber_outputs" / "runtime" / "curriculum.db"


class LlmSqlApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        main._provider = None
        self.db_patch = patch.object(main, "DEFAULT_CURRICULUM_DB_PATH", DB_PATH)
        self.db_patch.start()
        self._sql_service_patch = patch.object(main, "ask_sql", create=True)
        self.sql_service = self._sql_service_patch.start()
        self.sql_service.return_value = {
            "status": "answer",
            "answer": "พบ 4 วิชา",
            "sql": "SELECT ...",
            "columns": ["course_code"],
            "rows": [{"course_code": "C101"}],
        }

    def tearDown(self):
        self._sql_service_patch.stop()
        self.db_patch.stop()
        main._provider = None

    def test_api_ask_uses_sql_service_with_question_and_selected_program(self):
        with patch("rag.hybrid_demo.answer_question_once") as old_rag:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "ปี 3 เทอม 1 มีวิชาอะไรบ้าง",
                    "conversation_context": {"program": "IT", "catalog_key": "it-2565"},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "พบ 4 วิชา")
        self.assertEqual(response.json()["status"], "answer")
        self.assertEqual(response.json()["route"], "llm_sql")
        self.assertEqual(
            response.json()["next_context"], {"program": "IT", "catalog_key": "it-2565"}
        )
        self.assertNotIn("sql", response.json())
        self.assertNotIn("rows", response.json())
        self.assertEqual(self.sql_service.call_args.args[:3], (
            DB_PATH,
            "ปี 3 เทอม 1 มีวิชาอะไรบ้าง",
            "IT",
        ))
        self.assertTrue(callable(self.sql_service.call_args.args[3]))
        self.assertTrue(callable(self.sql_service.call_args.args[4]))
        old_rag.assert_not_called()

    def test_catalog_key_survives_api_result_context_round_trip(self):
        edition_context = {
            "program": "DSBA",
            "result_courses": [
                {"catalog_key": "dsba-2560", "program": "DSBA", "course_code": "C101"}
            ],
            "result_scope_program": "DSBA",
        }
        self.sql_service.return_value = {
            "status": "answer",
            "answer": "พบ 1 วิชา",
            "next_context": edition_context,
        }

        initial = self.client.post(
            "/api/ask",
            json={
                "question": "ปี 3 เทอม 1 มีวิชาอะไรบ้าง",
                "conversation_context": {
                    "program": "DSBA",
                    "catalog_key": "dsba-2560",
                },
            },
        )

        self.assertEqual(initial.status_code, 200)
        self.assertEqual(initial.json()["next_context"], edition_context)

        follow_up = self.client.post(
            "/api/ask",
            json={
                "question": "ในวิชาเหล่านี้มีกี่วิชา",
                "conversation_context": initial.json()["next_context"],
            },
        )

        self.assertEqual(follow_up.status_code, 200)
        forwarded = self.sql_service.call_args.kwargs["conversation_context"]
        self.assertEqual(
            forwarded["result_courses"][0]["catalog_key"], "dsba-2560"
        )

    def test_question_without_program_passes_none_and_returns_service_status(self):
        self.sql_service.return_value = {
            "status": "no_data",
            "answer": "ไม่พบข้อมูลที่ตรงกับคำถาม",
            "sql": "SELECT ...",
            "columns": ["course_code"],
            "rows": [],
        }

        response = self.client.post("/api/ask", json={"question": "รหัสวิชา X999 คืออะไร"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.sql_service.call_args.args[1:3], ("รหัสวิชา X999 คืออะไร", None))
        self.assertEqual(response.json()["status"], "no_data")
        self.assertEqual(response.json()["next_context"], None)

    def test_selected_catalog_is_validated_and_forwarded_to_sql(self):
        response = self.client.post(
            "/api/ask",
            json={
                "question": "DSBA มีวิชาอะไรบ้าง",
                "conversation_context": {
                    "program": "DSBA",
                    "catalog_key": "dsba-2560",
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["route"], "llm_sql")
        self.assertEqual(
            self.sql_service.call_args.kwargs["conversation_context"]["catalog_key"],
            "dsba-2560",
        )
        self.assertEqual(response.json()["next_context"]["catalog_key"], "dsba-2560")

    def test_selected_2565_catalog_is_forwarded_and_preserved_in_context(self):
        response = self.client.post(
            "/api/ask",
            json={
                "question": "DSBA มีวิชาอะไรบ้าง",
                "conversation_context": {
                    "program": "DSBA",
                    "catalog_key": "dsba-2565",
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        forwarded = self.sql_service.call_args.kwargs["conversation_context"]
        self.assertEqual(forwarded["catalog_key"], "dsba-2565")
        self.assertEqual(response.json()["next_context"]["catalog_key"], "dsba-2565")

    def test_dsba_total_requirement_with_edition_scope_answers(self):
        # Catalog-scoped requirements carry edition attribution through
        # their catalog row plus final-plan-page provenance, so an explicit
        # edition scope now answers (see test_rag_policy). Scopeless totals
        # still fail closed (test_multi_edition_dsba_total_requires_catalog_scope).
        for catalog_key, value in (("dsba-2560", "126"), ("dsba-2565", "132")):
            with self.subTest(catalog_key=catalog_key):
                self.sql_service.reset_mock()
                response = self.client.post(
                    "/api/ask",
                    json={
                        "question": "DSBA ต้องเรียนทั้งหมดกี่หน่วยกิต",
                        "conversation_context": {
                            "program": "DSBA",
                            "catalog_key": catalog_key,
                        },
                    },
                )

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["status"], "answer")
                self.assertIn(value, payload["answer"])
                self.assertNotIn("453", payload["answer"])
                self.assertNotIn("240", payload["answer"])
                self.assertTrue(payload["provenance"])
                self.sql_service.assert_not_called()

    def test_unscoped_multi_edition_question_is_stopped_before_any_qa_route(self):
        with patch.object(main, "answer_hard_question", return_value=None):
            response = self.client.post(
                "/api/ask",
                json={"question": "DSBA ปี 2 เทอม 1 เรียนกี่หน่วยกิต"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "clarification_required")
        self.assertEqual(response.json()["action"], "catalog_required")
        self.assertIn("dsba-2560", response.json()["answer"])
        self.assertIn("dsba-2565", response.json()["answer"])
        self.sql_service.assert_not_called()

    def test_explicit_it_query_returns_structural_context_and_real_provenance(self):
        def deterministic_sql_service(
            db_path, question, program, _sql_model, answer_model, *,
            conversation_context=None, **kwargs,
        ):
            grounding_callable = kwargs.pop("grounding_callable", None)
            return run_ask_sql(
                db_path,
                question,
                program,
                lambda _prompt: (
                    "SELECT course_code, program, plan_key, year, semester "
                    "FROM v_plan_courses"
                ),
                answer_model,
                conversation_context=conversation_context,
                grounding_callable=grounding_callable,
                **kwargs,
            )

        with patch.object(main, "answer_hard_question", return_value=None), patch.object(
            main, "ask_sql", side_effect=deterministic_sql_service
        ):
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "IT ปี 3 เทอม 1 มีทั้งหมดกี่วิชา",
                    "conversation_context": {"program": "IT", "catalog_key": "it-2565"},
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "answer")
        self.assertTrue(payload["provenance"])
        self.assertTrue(
            all(
                reference.get("program") == "IT"
                and reference.get("source_filename")
                and isinstance(reference.get("source_page"), int)
                for reference in payload["provenance"]
            )
        )
        context = payload["next_context"]
        self.assertEqual(context["program"], "IT")
        self.assertEqual(context["years"], [3])
        self.assertEqual(context["semesters"], [1])

    def test_it_semester_followup_keeps_program_and_year_scope(self):
        def deterministic_sql_service(
            db_path, question, program, _sql_model, answer_model, *,
            conversation_context=None, **kwargs,
        ):
            grounding_callable = kwargs.pop("grounding_callable", None)
            return run_ask_sql(
                db_path,
                question,
                program,
                lambda _prompt: (
                    "SELECT course_code, program, plan_key, year, semester "
                    "FROM v_plan_courses"
                ),
                answer_model,
                conversation_context=conversation_context,
                grounding_callable=grounding_callable,
                **kwargs,
            )

        with patch.object(main, "answer_hard_question", return_value=None), patch.object(
            main, "ask_sql", side_effect=deterministic_sql_service
        ):
            first = self.client.post(
                "/api/ask",
                json={
                    "question": "IT ปี 3 เทอม 1 มีทั้งหมดกี่วิชา",
                    "conversation_context": {"program": "IT", "catalog_key": "it-2565"},
                },
            )
            second = self.client.post(
                "/api/ask",
                json={
                    "question": "แล้วเทอม 2 ล่ะ",
                    "conversation_context": first.json()["next_context"],
                },
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        payload = second.json()
        self.assertEqual(payload["status"], "answer")
        self.assertTrue(payload["provenance"])
        self.assertTrue(all(ref.get("program") == "IT" for ref in payload["provenance"]))
        self.assertNotRegex(payload["answer"], r"(?i)\b(?:AIT|BIT|DSBA)\b")
        self.assertEqual(
            {
                key: payload["next_context"][key]
                for key in ("program", "years", "semesters")
            },
            {"program": "IT", "years": [3], "semesters": [2]},
        )

    def test_dsba_edition_scoped_followup_keeps_catalog_and_real_provenance(self):
        def deterministic_sql_service(
            db_path, question, program, _sql_model, answer_model, *,
            conversation_context=None, **kwargs,
        ):
            return run_ask_sql(
                db_path,
                question,
                program,
                lambda _prompt: (
                    "SELECT course_code, program, plan_key, year, semester "
                    "FROM v_plan_courses"
                ),
                answer_model,
                conversation_context=conversation_context,
                **kwargs,
            )

        outputs = {}
        with patch.object(main, "answer_hard_question", return_value=None), patch.object(
            main, "ask_sql", side_effect=deterministic_sql_service
        ):
            for catalog_key in ("dsba-2560", "dsba-2565"):
                response = self.client.post(
                    "/api/ask",
                    json={
                        "question": "แล้วเทอม 2 ล่ะ",
                        "conversation_context": {
                            "program": "DSBA",
                            "catalog_key": catalog_key,
                            "years": [2],
                            "semesters": [1],
                            "operations": ["count"],
                        },
                    },
                )
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["status"], "answer")
                self.assertTrue(payload["provenance"])
                self.assertTrue(
                    all(
                        reference.get("source_filename")
                        and isinstance(reference.get("source_page"), int)
                        for reference in payload["provenance"]
                    )
                )
                self.assertEqual(payload["next_context"]["program"], "DSBA")
                self.assertEqual(payload["next_context"]["catalog_key"], catalog_key)
                self.assertEqual(payload["next_context"]["years"], [2])
                self.assertEqual(payload["next_context"]["semesters"], [2])
                outputs[catalog_key] = payload["provenance"]

        self.assertNotEqual(
            {item["source_filename"] for item in outputs["dsba-2560"]},
            {item["source_filename"] for item in outputs["dsba-2565"]},
        )

    def test_semester_followup_without_context_fails_before_sql(self):
        with patch.object(main, "answer_hard_question", return_value=None):
            response = self.client.post(
                "/api/ask", json={"question": "แล้วเทอม 2 ล่ะ"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.json()["status"], "answer")
        self.assertIn(response.json()["action"], {"program_required", "clarify_program"})
        self.assertEqual(response.json()["provenance"], [])
        self.assertIsNone(response.json()["next_context"])
        self.sql_service.assert_not_called()

    def test_explicit_program_replaces_prior_program_context(self):
        with patch.object(main, "answer_hard_question", return_value=None):
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "แล้ว DSBA ปี 1 เทอม 1 มีกี่วิชา",
                    "conversation_context": {
                        "program": "IT",
                        "years": [3],
                        "semesters": [2],
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "clarification_required")
        self.assertEqual(response.json()["action"], "catalog_required")
        self.assertIn("dsba-2560", response.json()["answer"])
        self.assertIn("dsba-2565", response.json()["answer"])
        self.sql_service.assert_not_called()

    def test_year_semester_focus_is_forwarded_as_structural_sql_context(self):
        response = self.client.post(
            "/api/ask",
            json={
                "question": "แล้วมีวิชาอะไรบ้าง",
                "conversation_context": {
                    "program": "DSBA",
                    "catalog_key": "dsba-2565",
                    "years": [2],
                    "semesters": [1],
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        forwarded = self.sql_service.call_args.kwargs["conversation_context"]
        self.assertEqual(forwarded["catalog_key"], "dsba-2565")
        self.assertEqual(forwarded["years"], [2])
        self.assertEqual(forwarded["semesters"], [1])

    def test_edition_switch_clears_stale_focus_and_result_context_both_directions(self):
        for old_key, new_key in (
            ("dsba-2565", "dsba-2560"),
            ("dsba-2560", "dsba-2565"),
        ):
            with self.subTest(old_key=old_key, new_key=new_key):
                self.sql_service.reset_mock()
                response = self.client.post(
                    "/api/ask",
                    json={
                        "question": "DSBA ปี 1 เทอม 1 มีกี่วิชา",
                        "conversation_context": {
                            "program": "DSBA",
                            "catalog_key": new_key,
                            "focus_course": {
                                "catalog_key": old_key,
                                "program": "DSBA",
                                "course_code": "06026111",
                            },
                            "plan": "coop",
                            "years": [2],
                            "semesters": [1],
                            "result_scope_program": "DSBA",
                            "result_courses": [
                                {
                                    "catalog_key": old_key,
                                    "program": "DSBA",
                                    "course_code": "06026111",
                                }
                            ],
                        },
                    },
                )

                self.assertEqual(response.status_code, 200)
                forwarded = self.sql_service.call_args.kwargs["conversation_context"]
                self.assertEqual(forwarded["catalog_key"], new_key)
                self.assertEqual(forwarded["focus_catalog_key"], new_key)
                self.assertNotIn("years", forwarded)
                self.assertNotIn("semesters", forwarded)
                self.assertNotIn("plan", forwarded)
                self.assertNotIn("result_courses", forwarded)
                self.assertNotIn("result_scope_program", forwarded)

    def test_nonexistent_catalog_key_is_rejected_by_api(self):
        response = self.client.post(
            "/api/ask",
            json={
                "question": "DSBA มีวิชาอะไรบ้าง",
                "conversation_context": {
                    "program": "DSBA",
                    "catalog_key": "dsba-9999",
                },
            },
        )

        self.assertEqual(response.status_code, 422)
        self.sql_service.assert_not_called()

    def test_controlled_service_failure_maps_to_existing_response_contract(self):
        self.sql_service.return_value = {
            "status": "error",
            "answer": "",
            "sql": "",
            "columns": [],
            "rows": [],
            "error": {"code": "invalid_sql"},
        }

        response = self.client.post("/api/ask", json={"question": "แสดงวิชา"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "error")
        self.assertEqual(response.json()["action"], "invalid_sql")
        self.assertEqual(response.json()["answer"], "")

    def test_provider_unavailability_remains_http_503(self):
        self.sql_service.side_effect = main.ProviderUnavailable("provider unavailable")
        response = self.client.post("/api/ask", json={"question": "แสดงวิชา"})

        self.assertEqual(response.status_code, 503)

    def test_programs_endpoint_remains_available(self):
        response = self.client.get("/api/programs")

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json()["programs"], list)
        dsba = next(item for item in response.json()["programs"] if item["program_code"] == "DSBA")
        self.assertEqual(
            {(edition["catalog_key"], edition["academic_year"]) for edition in dsba["editions"]},
            {("dsba-2560", "2560"), ("dsba-2565", "2565")},
        )

    def test_curriculum_endpoint_filters_by_catalog(self):
        response = self.client.get(
            "/api/curriculum",
            params={"program": "DSBA", "catalog_key": "dsba-2560", "limit": 5},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["items"])
        self.assertTrue(
            all(
                item["catalog_key"] == "dsba-2560" and item["academic_year"] == "2560"
                for item in response.json()["items"]
            )
        )

    def test_curriculum_endpoint_rejects_unknown_catalog(self):
        response = self.client.get(
            "/api/curriculum", params={"catalog_key": "dsba-9999"}
        )

        self.assertEqual(response.status_code, 422)

    def test_course_detail_resolves_shared_edition_row_without_artificial_split(self):
        # Frozen contract: one shared course row per edition; plan/source
        # provenance does not split it. Obsolete source-string catalog keys
        # are invalid identities.
        resolved = self.client.get("/api/courses/06016401", params={"program": "IT"})
        selected = self.client.get(
            "/api/courses/06016401",
            params={
                "program": "IT",
                "catalog_key": "it-2565",
            },
        )

        self.assertEqual(resolved.status_code, 200)
        self.assertEqual(resolved.json()["course"]["catalog_key"], "it-2565")
        self.assertEqual(selected.status_code, 200)
        self.assertEqual(
            selected.json()["course"]["catalog_key"],
            "it-2565",
        )
        stale = self.client.get(
            "/api/courses/06016401",
            params={
                "program": "IT",
                "catalog_key": "OCR extraction / Academic Plan - IT no_coop",
            },
        )
        self.assertEqual(stale.status_code, 422)

    def test_dsba_course_details_resolve_inside_selected_edition(self):
        cases = (
            ("06026100", "dsba-2560", "2560"),
            ("06066300", "dsba-2565", "2565"),
        )
        for course_code, catalog_key, academic_year in cases:
            with self.subTest(catalog_key=catalog_key, course_code=course_code):
                response = self.client.get(
                    f"/api/courses/{course_code}",
                    params={"program": "DSBA", "catalog_key": catalog_key},
                )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["course"]["catalog_key"], catalog_key)
                self.assertEqual(response.json()["course"]["academic_year"], academic_year)

    def test_curriculum_endpoint_remains_available(self):
        response = self.client.get("/api/curriculum", params={"program": "IT", "limit": 1})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["limit"], 1)
        self.assertIsInstance(response.json()["items"], list)

    def test_hard_comparison_uses_deterministic_h1_and_returns_provenance(self):
        question = "DSBA coop กับ no_coop ต่างกันที่วิชาไหน"
        context = {"program": "DSBA", "catalog_key": "dsba-2565"}
        intent = {
            "task_type": "plan_comparison",
            "program": "DSBA",
            "plan": None,
            "left_plan": "coop",
            "right_plan": "no_coop",
            "target_course_code": None,
            "horizon_terms": None,
        }
        expected = answer_hard_question(DB_PATH, question, context, lambda prompt: json.dumps(intent))
        with patch.object(main, "_lazy_provider", return_value=json.dumps(intent)) as provider:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": question,
                    "conversation_context": context,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["route"], "hard")
        self.assertEqual(payload["hard_task_type"], "plan_comparison")
        self.assertEqual(payload["status"], "answer")
        self.assertIn("ไม่พบความแตกต่างของชุดรหัสวิชา", payload["answer"])
        self.assertTrue(payload["provenance"])
        self.assertIn("อ้างอิง:", payload["answer"])
        self.assertEqual(payload["provenance"], expected["provenance"])
        self.assertTrue(all("source_filename" in item and "document_page" in item for item in payload["provenance"]))
        self.assertEqual(self.sql_service.call_count, 0)
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(provider.call_args.kwargs["response_mime_type"], "application/json")
        self.assertEqual(provider.call_args.kwargs["response_json_schema"], HARD_INTERPRETATION_RESPONSE_JSON_SCHEMA)
        self.assertNotIn("response_schema", provider.call_args.kwargs)

    def test_old_new_comparison_api_returns_structured_categories_without_model(self):
        with patch.object(
            main, "_lazy_provider", side_effect=AssertionError("comparison is deterministic")
        ) as provider:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "วิชาที่มีในหลักสูตรเก่า DSBA ไม่มีในหลักสูตรใหม่มีอะไรบ้าง",
                    "conversation_context": {"program": "DSBA"},
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["route"], "hard")
        self.assertEqual(payload["hard_task_type"], "old_new_comparison")
        self.assertEqual(payload["comparison"]["older"]["catalog_key"], "dsba-2560")
        self.assertEqual(payload["comparison"]["newer"]["catalog_key"], "dsba-2565")
        self.assertIn("same_name_changed_code_candidates", payload["comparison"]["categories"])
        self.assertIn("หลักสูตร DSBA พ.ศ. 2560", payload["answer"])
        self.assertIn("ตัวเลือกที่อาจเป็นการเปลี่ยนรหัสวิชา", payload["answer"])
        self.assertIn("ยังไม่ถือว่าเป็นการยืนยัน", payload["answer"])
        self.assertTrue(payload["provenance"])
        candidate = payload["comparison"]["categories"]["same_name_changed_code_candidates"][0]
        self.assertFalse(candidate["equivalence_proven"])
        source_names = {item["source_filename"] for item in payload["comparison"]["provenance"]}
        self.assertTrue(any(name.startswith("dsba2560_") for name in source_names))
        self.assertTrue(any(name.startswith("dsba_") for name in source_names))
        provider.assert_not_called()

    def test_old_new_comparison_api_preserves_plan_and_both_edition_provenance(self):
        with patch.object(
            main, "_lazy_provider", side_effect=AssertionError("comparison is deterministic")
        ) as provider:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "วิชาใดในหลักสูตรเก่า DSBA coop ไม่พบในหลักสูตรใหม่",
                    "conversation_context": {"program": "DSBA"},
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["route"], "hard")
        self.assertEqual(payload["hard_task_type"], "old_new_comparison")
        comparison = payload["comparison"]
        self.assertEqual(comparison["older"], {"catalog_key": "dsba-2560", "academic_year": "2560"})
        self.assertEqual(comparison["newer"], {"catalog_key": "dsba-2565", "academic_year": "2565"})
        self.assertEqual(comparison["plan"], "coop")
        self.assertIn("ขอบเขตแผน: coop", payload["answer"])
        for category in (
            "shared_same_code", "old_only_by_code", "new_only_by_code",
            "same_name_changed_code_candidates",
        ):
            for bucket in comparison["categories"][category]:
                courses = bucket.get("courses", []) or [
                    bucket.get("older", {}), bucket.get("newer", {})
                ]
                for course in courses:
                    if course:
                        self.assertEqual(course["plans"], ["coop"])
        source_names = {item["source_filename"] for item in comparison["provenance"]}
        self.assertTrue(any(name.startswith("dsba2560_") for name in source_names))
        self.assertTrue(any(name.startswith("dsba_") for name in source_names))
        self.assertTrue(payload["provenance"])
        self.assertIn("ยังไม่ถือว่าเป็นการยืนยัน", payload["answer"])
        provider.assert_not_called()

    def test_rejected_hard_interpretation_does_not_fall_through_to_sql_service(self):
        with patch.object(main, "_lazy_provider", return_value="not JSON") as provider:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "DSBA แผน coop กับ no_coop ต่างกันที่วิชาไหนบ้าง",
                    "conversation_context": {"program": "DSBA"},
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["route"], "hard")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["action"], "hard_interpretation_failure")
        self.assertIn("ยังจัดประเภทคำถาม Hard นี้ไม่ได้อย่างปลอดภัย", payload["answer"])
        provider.assert_called_once()
        self.assertEqual(provider.call_args.kwargs["response_mime_type"], "application/json")
        self.assertEqual(provider.call_args.kwargs["response_json_schema"], HARD_INTERPRETATION_RESPONSE_JSON_SCHEMA)
        self.assertNotIn("response_schema", provider.call_args.kwargs)
        self.sql_service.assert_not_called()


if __name__ == "__main__":
    unittest.main()
