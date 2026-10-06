import unittest
import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main
from backend.hard_qa import HARD_INTERPRETATION_RESPONSE_JSON_SCHEMA, answer_hard_question
from backend.llm_sql_qa import ask_sql as run_ask_sql
from backend.llm_sql_qa import _retained_identities_from_answer
from rag import qa as rag_qa
from rag.query_spec import detect_surface_operations, parse_query_spec


DB_PATH = Path(__file__).parents[1] / "cucumber_outputs" / "runtime" / "curriculum.db"


class LlmSqlApiTests(unittest.TestCase):
    def test_missing_plan_has_structured_action_and_selection_reaches_plan_scope(self):
        proposal = {"task_type": "prerequisite_sequence", "program": "IT", "plan": None,
                    "left_plan": None, "right_plan": None, "target_course_code": None, "horizon_terms": None}
        with patch.object(main, "_lazy_provider", return_value=json.dumps(proposal)):
            missing = self.client.post("/api/ask", json={
                "question": "ตรวจสอบลำดับวิชาบังคับก่อนของแผนนี้",
                "conversation_context": {"program": "IT", "catalog_key": "it-2565"},
            })
            self.assertEqual(missing.status_code, 200)
            self.assertEqual(missing.json()["status"], "clarification_required")
            self.assertEqual(missing.json()["action"], "plan_required")
            for plan in ("coop", "no_coop"):
                selected = self.client.post("/api/ask", json={
                    "question": "ตรวจสอบลำดับวิชาบังคับก่อนของแผนนี้",
                    "conversation_context": {"program": "IT", "catalog_key": "it-2565", "plan": plan},
                })
                self.assertEqual(selected.status_code, 200)
                self.assertNotEqual(selected.json().get("action"), "plan_required")
                self.assertEqual(selected.json()["next_context"]["plan"], plan)

    def test_fresh_gpa_probation_value_comparison_is_deterministic_policy_answer(self):
        self.sql_service.side_effect = lambda *args, **kwargs: run_ask_sql(*args, **kwargs)

        def ask_without_model(question):
            with (
                patch.object(main, "answer_hard_question", return_value=None),
                patch.object(main, "_lazy_provider", side_effect=AssertionError("policy comparison must be deterministic")),
            ):
                response = self.client.post("/api/ask", json={"question": question})
            self.assertEqual(response.status_code, 200, response.json())
            return response.json()

        for question, included in (
            ("GPA 1.9 จะติดโปรไหม", "เข้าข่ายภาคทัณฑ์"),
            ("ถ้าเกรดเฉลี่ยสะสม 1.9 จะเข้าข่ายภาคทัณฑ์ไหม", "เข้าข่ายภาคทัณฑ์"),
            ("GPA 1.9 เข้าข่ายโปรหรือเปล่า", "เข้าข่ายภาคทัณฑ์"),
        ):
            with self.subTest(question=question):
                payload = ask_without_model(question)
                self.assertEqual(payload["status"], "answer")
                self.assertIn(included, payload["answer"])
                self.assertIn("1.9", payload["answer"])
                self.assertTrue(payload["provenance"])

        repeated = ask_without_model("GPA 1.9 จะติดโปรไหม")
        self.assertEqual(repeated["status"], "answer")
        self.assertIn("1.9", repeated["answer"])

        for question in ("GPA จะติดโปรไหม", "GPA 1..9 จะติดโปรไหม"):
            with self.subTest(question=question):
                malformed = ask_without_model(question)
                self.assertEqual(malformed["status"], "unsupported")
                self.assertEqual(malformed["provenance"], [])

    def test_prior_relationship_result_set_does_not_block_current_hard_sequence(self):
        context = {"program": "AIT", "catalog_key": "ait-2566", "plan": "default"}
        hard_intent = {
            "task_type": "prerequisite_sequence",
            "program": "AIT",
            "plan": "default",
            "left_plan": None,
            "right_plan": None,
            "target_course_code": None,
            "horizon_terms": None,
        }
        sequence_question = "ตรวจสอบลำดับวิชาบังคับก่อนของหลักสูตรนี้"
        with patch.object(main, "_lazy_provider", return_value=json.dumps(hard_intent)):
            fresh = self.client.post(
                "/api/ask",
                json={"question": sequence_question, "conversation_context": context},
            )
        self.assertEqual(fresh.status_code, 200, fresh.json())
        self.assertEqual(fresh.json()["route"], "hard")
        self.assertEqual(fresh.json()["hard_task_type"], "prerequisite_sequence")
        self.assertEqual(fresh.json()["status"], "violation")

        relationship = self._ask_with_stub_provider(
            "วิชาใดบ้างที่มีวิชาบังคับก่อน บอกชื่อและรหัสวิชามา", context
        )
        self.assertEqual(relationship["status"], "answer")
        self.assertEqual(relationship["route"], "llm_sql")
        self.assertTrue(relationship["provenance"])
        self.assertTrue(relationship["next_context"]["result_courses"])

        with patch.object(main, "_lazy_provider", return_value=json.dumps(hard_intent)):
            response = self.client.post(
                "/api/ask",
                json={
                    "question": sequence_question,
                    "conversation_context": relationship["next_context"],
                },
            )
        self.assertEqual(response.status_code, 200, response.json())
        payload = response.json()
        self.assertEqual(payload["route"], "hard")
        self.assertEqual(payload["hard_task_type"], "prerequisite_sequence")
        self.assertEqual(payload["status"], "violation")
        self.assertTrue(payload["provenance"])
        self.assertEqual(payload["next_context"]["program"], "AIT")
        self.assertEqual(payload["next_context"]["catalog_key"], "ait-2566")
        self.assertEqual(payload["next_context"]["plan"], "default")

    def test_semantic_existence_answers_are_presented_in_natural_language(self):
        scope = {"program": "IT", "catalog_key": "it-2565"}
        for question in (
            "มีวิชาเกี่ยวกับ cloud มั้ย",
            "มีวิชาที่เกี่ยวกับ cloud ไหม",
        ):
            with self.subTest(question=question):
                payload = self._ask_with_stub_provider(question, scope)
                self.assertEqual(payload["status"], "answer")
                self.assertNotIn("existence:", payload["answer"])
                self.assertIn("มีครับ", payload["answer"])
                self.assertTrue(payload["provenance"])

    def test_semantic_collection_wording_still_renders_course_list(self):
        payload = self._ask_with_stub_provider(
            "มีวิชาเกี่ยวกับ cloud อะไรบ้าง", {"program": "IT", "catalog_key": "it-2565"}
        )
        self.assertEqual(payload["status"], "answer")
        self.assertNotIn("existence:", payload["answer"])
        self.assertIn("06016404", payload["answer"])

    def test_h4_term_credit_followups_use_retained_canonical_structure(self):
        h4_intent = {
            "task_type": "seven_term_plan", "program": "DSBA", "plan": "coop",
            "left_plan": None, "right_plan": None, "target_course_code": None,
            "horizon_terms": 7,
        }
        with patch.object(main, "_lazy_provider", return_value=json.dumps(h4_intent)):
            first_response = self.client.post("/api/ask", json={
                "question": "อยากจบใน 3.5 ปีต้องทำยังไง",
                "conversation_context": {"program": "DSBA", "catalog_key": "dsba-2565", "plan": "coop"},
            })
        self.assertEqual(first_response.status_code, 200, first_response.json())
        first = first_response.json()
        self.assertEqual(first["hard_task_type"], "seven_term_plan")
        self.assertTrue(first["provenance"])
        context = first["next_context"]
        self.assertEqual(context.get("study_plan_context", {}).get("plan"), "coop")

        full = self.client.post("/api/ask", json={
            "question": "ปี 1 เทอม 1 หน่วยกิตเท่าไหร่", "conversation_context": context,
        }).json()
        self.assertEqual(full["status"], "answer")
        self.assertIn("ปี 1 เทอม 1 รวม 18 หน่วยกิต", full["answer"])
        self.assertNotIn("06026200", full["answer"])
        self.assertTrue(full["provenance"])

        partial = self.client.post("/api/ask", json={
            "question": "ปี 3 เทอม 1 รวมกี่หน่วยกิต", "conversation_context": context,
        }).json()
        self.assertEqual(partial["status"], "answer")
        self.assertIn("ยืนยันได้อย่างน้อย 9 หน่วยกิต", partial["answer"])
        self.assertIn("ยังสรุปหน่วยกิตรวมทั้งหมดไม่ได้", partial["answer"])
        self.assertIn("3 ช่อง", partial["answer"])
        self.assertNotIn("รวม 9 หน่วยกิต", partial["answer"])
        self.assertTrue(partial["provenance"])

        course_list = self.client.post("/api/ask", json={
            "question": "ปี 1 เทอม 1 มีวิชาอะไรบ้าง", "conversation_context": context,
        }).json()
        self.assertEqual(course_list["status"], "answer")
        self.assertIn("06026200", course_list["answer"])
        full_count = len(first["provenance"])
        for label, payload in (
            ("term credit", full),
            ("partial term credit", partial),
            ("term course list", course_list),
        ):
            with self.subTest(label=label):
                self.assertTrue(payload["provenance"])
                self.assertLess(len(payload["provenance"]), full_count)

        invalid = self.client.post("/api/ask", json={
            "question": "ปี 9 เทอม 4 หน่วยกิตเท่าไหร่", "conversation_context": context,
        }).json()
        self.assertEqual(invalid["status"], "insufficient_evidence")
        self.assertEqual(invalid["provenance"], [])

        fresh_context = {key: value for key, value in context.items() if key != "study_plan_context"}
        fresh = self.client.post("/api/ask", json={
            "question": "ปี 1 เทอม 1 หน่วยกิตเท่าไหร่", "conversation_context": fresh_context,
        }).json()
        self.assertNotEqual(fresh.get("hard_task_type"), "seven_term_plan")
        self.assertNotIn("study_plan_context", fresh.get("next_context") or {})

    def test_h4_term_followup_fails_closed_without_matching_prior_plan_scope(self):
        question = "ปี 1 เทอม 1 หน่วยกิตเท่าไหร่"
        for context in (
            {"program": "DSBA", "catalog_key": "dsba-2565", "plan": "coop"},
            {"program": "DSBA", "catalog_key": "dsba-2565", "plan": "coop",
             "study_plan_context": {"program": "DSBA", "catalog_key": "dsba-2565", "plan": "no_coop"}},
            {"program": "IT", "catalog_key": "it-2565", "plan": "coop",
             "study_plan_context": {"program": "DSBA", "catalog_key": "dsba-2565", "plan": "coop"}},
        ):
            with self.subTest(context=context):
                response = self.client.post("/api/ask", json={"question": question, "conversation_context": context})
                self.assertEqual(response.status_code, 200, response.json())
                payload = response.json()
                self.assertNotEqual(payload.get("hard_task_type"), "seven_term_plan")
                self.assertNotIn("study_plan_context", payload.get("next_context") or {})

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

    def test_retake_timing_question_fails_closed_without_offering_data(self):
        question = (
            "ถ้าถอนวิชา FUNDAMENTAL WEB PROGRAMMING ตอนปี 2 เทอม 1 "
            "ต้องลงเรียนอีกทีตอนไหน เทอมไหน"
        )
        contexts = (
            {"program": "DSBA", "catalog_key": "dsba-2565"},
            {
                "program": "DSBA",
                "catalog_key": "dsba-2565",
                "plan": "no_coop",
                "years": [2],
                "semesters": [1],
                "operations": ["placement"],
            },
        )
        with (
            patch.object(main, "answer_hard_question") as hard_qa,
            patch.object(main, "_lazy_provider") as provider,
        ):
            for context in contexts:
                response = self.client.post(
                    "/api/ask",
                    json={"question": question, "conversation_context": context},
                )
                self.assertEqual(response.status_code, 200, response.json())
                payload = response.json()
                self.assertEqual(payload["status"], "insufficient_evidence")
                self.assertEqual(payload["action"], "course_offering_data_required")
                self.assertEqual(payload["route"], "llm_sql")
                self.assertEqual(payload["provenance"], [])
                self.assertIn("ไม่มีข้อมูลการเปิดสอนจริง", payload["answer"])
                self.assertIn("ระบบลงทะเบียนของมหาวิทยาลัย", payload["answer"])
                next_context = payload["next_context"]
                self.assertEqual(next_context["program"], "DSBA")
                self.assertEqual(next_context["catalog_key"], "dsba-2565")
                self.assertEqual(
                    next_context.get("plan"), context.get("plan")
                )
                for key in (
                    "years",
                    "semesters",
                    "operations",
                    "result_courses",
                    "focus_course",
                ):
                    self.assertNotIn(key, next_context)
        self.sql_service.assert_not_called()
        hard_qa.assert_not_called()
        provider.assert_not_called()

    def test_retake_timing_guard_does_not_capture_supported_question_families(self):
        questions = (
            "06066300 เรียนปีไหน เทอมไหน",
            "FUNDAMENTAL WEB PROGRAMMING อยู่ปีไหน",
            "ถอนรายวิชาได้ถึงเมื่อไหร่",
            "06066300 ต้องเรียนอะไรมาก่อน",
            "DSBA ปี 2 เทอม 1 มีวิชาอะไรบ้าง",
        )
        with (
            patch.object(main, "answer_hard_question", return_value=None),
            patch.object(main, "_lazy_provider", side_effect=AssertionError("unexpected model call")),
        ):
            for question in questions:
                with self.subTest(question=question):
                    response = self.client.post(
                        "/api/ask",
                        json={
                            "question": question,
                            "conversation_context": {
                                "program": "DSBA",
                                "catalog_key": "dsba-2565",
                            },
                        },
                    )
                    self.assertEqual(response.status_code, 200, response.json())
                    self.assertNotEqual(
                        response.json().get("action"),
                        "course_offering_data_required",
                    )

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
        self.assertNotIn("plan_results", response.json())
        self.assertEqual(self.sql_service.call_args.args[:3], (
            DB_PATH,
            "ปี 3 เทอม 1 มีวิชาอะไรบ้าง",
            "IT",
        ))
        self.assertTrue(callable(self.sql_service.call_args.args[3]))
        self.assertTrue(callable(self.sql_service.call_args.args[4]))
        old_rag.assert_not_called()

    def test_contextual_positive_prerequisite_collection_returns_grounded_api_answer(self):
        def deterministic_grounding(
            _db_path,
            question,
            _program,
            _structured_provider,
            _answer_provider,
            *,
            conversation_context,
            grounding_callable,
        ):
            grounded = grounding_callable(question)
            return {
                "status": grounded["status"],
                "answer": grounded["final_answer"],
                "provenance": list(grounded["provenance"]),
                "next_context": conversation_context,
            }

        self.sql_service.side_effect = deterministic_grounding
        with patch.object(main, "answer_hard_question", return_value=None):
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "วิชาใดมีวิชาบังคับก่อนบ้าง บอกชื่อและรหัสวิชามา",
                    "conversation_context": {
                        "program": "DSBA",
                        "catalog_key": "dsba-2565",
                        "operations": ["list"],
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "answer")
        self.assertEqual(payload["route"], "llm_sql")
        self.assertEqual(
            payload["next_context"],
            {"program": "DSBA", "catalog_key": "dsba-2565", "operations": ["list"]},
        )
        self.assertTrue(payload["provenance"])
        for course_code in ("06026201", "06026212", "06026213", "06026215", "06066102"):
            self.assertIn(course_code, payload["answer"])
        self.assertIn("สรุปว่าไม่มีวิชาบังคับก่อนไม่ได้", payload["answer"])

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

    def test_catalog_clarification_followup_resumes_original_term_list(self):
        def real_sql_service(
            db_path,
            question,
            program,
            _sql_model,
            answer_model,
            *,
            conversation_context=None,
            grounding_callable=None,
            **kwargs,
        ):
            return run_ask_sql(
                db_path,
                question,
                program,
                lambda _prompt: (
                    "SELECT course_code, program, plan_key, year, semester "
                    "FROM v_plan_courses"
                ),
                lambda _prompt: self.fail("grounded results must not be synthesized"),
                conversation_context=conversation_context,
                grounding_callable=grounding_callable,
                **kwargs,
            )

        self.sql_service.side_effect = real_sql_service
        with patch.object(main, "_lazy_provider", return_value="unused"):
            first = self.client.post(
                "/api/ask",
                json={"question": "วิชาที่เรียนในปี 1 เทอม 1 ของหลักสูตร DSBA"},
            )
            self.assertEqual(first.status_code, 200)
            first_payload = first.json()
            self.assertEqual(first_payload["action"], "catalog_required")
            pending = first_payload["next_context"]
            self.assertEqual(pending["program"], "DSBA")
            self.assertEqual(pending["operations"], ["list"])
            self.assertEqual(pending["years"], [1])
            self.assertEqual(pending["semesters"], [1])
            self.assertIs(pending["pending_catalog_selection"], True)

            for reply, catalog_key, expected_code, excluded_code in (
                ("ของปี 2565", "dsba-2565", "06026200", "06026100"),
                ("2560", "dsba-2560", "06026100", "06026200"),
            ):
                with self.subTest(reply=reply):
                    response = self.client.post(
                        "/api/ask",
                        json={
                            "question": reply,
                            "conversation_context": pending,
                        },
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    payload = response.json()
                    self.assertEqual(payload["status"], "answer", payload)
                    self.assertIn("หลักสูตร DSBA", payload["answer"])
                    self.assertIn("ปี 1 ภาคเรียนที่ 1", payload["answer"])
                    self.assertIn(expected_code, payload["answer"])
                    self.assertNotIn(excluded_code, payload["answer"])
                    self.assertTrue(payload["provenance"])
                    resumed = payload["next_context"] or {}
                    self.assertEqual(resumed.get("program"), "DSBA")
                    self.assertEqual(resumed.get("catalog_key"), catalog_key)
                    self.assertEqual(resumed.get("years"), [1])
                    self.assertEqual(resumed.get("semesters"), [1])
                    self.assertEqual(resumed.get("operations"), ["list"])
                    # Micro-task 8: the answered list additionally retains its
                    # canonical identities for ordinal follow-ups.
                    retained = resumed.get("result_courses") or []
                    self.assertTrue(retained)
                    for course in retained:
                        self.assertEqual(course.get("program"), "DSBA")
                        self.assertEqual(course.get("catalog_key"), catalog_key)

    def test_invalid_pending_edition_stays_in_clarification(self):
        first = self.client.post(
            "/api/ask",
            json={"question": "วิชาที่เรียนในปี 1 เทอม 1 ของหลักสูตร DSBA"},
        )
        pending = first.json()["next_context"]

        for reply in ("ของปี 2999", "IT ของปี 2565"):
            with self.subTest(reply=reply):
                response = self.client.post(
                    "/api/ask",
                    json={"question": reply, "conversation_context": pending},
                )

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["status"], "clarification_required")
                self.assertEqual(payload["action"], "catalog_required")
                self.assertEqual(payload["next_context"], pending)
                self.assertEqual(payload["provenance"], [])
        self.sql_service.assert_not_called()

    def test_single_turn_study_year_is_preserved_as_study_scope(self):
        response = self.client.post(
            "/api/ask",
            json={"question": "DSBA ปี 1 เทอม 1 มีวิชาอะไรบ้าง"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["action"], "catalog_required")
        self.assertEqual(payload["next_context"]["years"], [1])
        self.assertEqual(payload["next_context"]["semesters"], [1])
        self.assertEqual(payload["next_context"]["operations"], ["list"])
        self.assertNotIn("catalog_key", payload["next_context"])
        self.sql_service.assert_not_called()

    def test_exact_course_answers_are_isolated_to_selected_catalog(self):
        captured = []

        def real_sql_service(
            db_path,
            question,
            program,
            _sql_model,
            _answer_model,
            *,
            conversation_context=None,
            grounding_callable=None,
            **kwargs,
        ):
            course_code = question[:8]
            result = run_ask_sql(
                db_path,
                question,
                program,
                lambda _prompt: (
                    "SELECT course_code, program, plan_key, year, semester "
                    f"FROM v_plan_courses WHERE course_code = '{course_code}'"
                ),
                lambda _prompt: self.fail("grounded results must not be synthesized"),
                conversation_context=conversation_context,
                grounding_callable=grounding_callable,
                **kwargs,
            )
            captured.append(result)
            return result

        self.sql_service.side_effect = real_sql_service
        cases = (
            ("dsba-2560", "06026212", "กี่หน่วยกิต", False),
            ("dsba-2560", "06026200", "กี่หน่วยกิต", False),
            ("dsba-2565", "06026212", "กี่หน่วยกิต", True),
            ("dsba-2565", "06026212", "เรียนเกี่ยวกับอะไร", True),
            ("dsba-2565", "06026212", "เรียนปีไหน", True),
        )
        for catalog_key, course_code, wording, should_answer in cases:
            with self.subTest(catalog_key=catalog_key, course_code=course_code, wording=wording):
                response = self.client.post(
                    "/api/ask",
                    json={
                        "question": f"{course_code} {wording}",
                        "conversation_context": {
                            "program": "DSBA",
                            "catalog_key": catalog_key,
                        },
                    },
                )

                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()
                sql_result = captured[-1]
                self.assertEqual(sql_result["sql"].count("LIMIT"), 1)
                if should_answer:
                    self.assertEqual(payload["status"], "answer", payload)
                    self.assertTrue(payload["provenance"])
                    self.assertEqual(payload["next_context"]["catalog_key"], catalog_key)
                    self.assertEqual(payload["next_context"]["course_code"], course_code)
                    if wording == "กี่หน่วยกิต":
                        self.assertIn("3 หน่วยกิต", payload["answer"])
                else:
                    self.assertNotEqual(payload["status"], "answer", payload)
                    self.assertEqual(payload["provenance"], [])
                    next_context = payload.get("next_context") or {}
                    if next_context:
                        self.assertEqual(next_context.get("catalog_key"), catalog_key)
                    self.assertNotEqual(next_context.get("course_code"), course_code)
                    self.assertEqual(sql_result["rows"], [])

    def test_exact_course_term_lists_remain_edition_isolated(self):
        def real_sql_service(
            db_path,
            question,
            program,
            _sql_model,
            _answer_model,
            *,
            conversation_context=None,
            grounding_callable=None,
            **kwargs,
        ):
            return run_ask_sql(
                db_path,
                question,
                program,
                lambda _prompt: (
                    "SELECT course_code, program, plan_key, year, semester "
                    "FROM v_plan_courses"
                ),
                lambda _prompt: self.fail("grounded results must not be synthesized"),
                conversation_context=conversation_context,
                grounding_callable=grounding_callable,
                **kwargs,
            )

        self.sql_service.side_effect = real_sql_service
        answers = {}
        for catalog_key in ("dsba-2560", "dsba-2565"):
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "DSBA ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
                    "conversation_context": {
                        "program": "DSBA",
                        "catalog_key": catalog_key,
                    },
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["status"], "answer", payload)
            self.assertTrue(payload["provenance"])
            self.assertEqual(payload["next_context"]["catalog_key"], catalog_key)
            answers[catalog_key] = payload["answer"]

        self.assertIn("06026100", answers["dsba-2560"])
        self.assertNotIn("06026200", answers["dsba-2560"])
        self.assertIn("06026200", answers["dsba-2565"])
        self.assertNotIn("06026100", answers["dsba-2565"])

    def test_topic_describe_returns_scoped_canonical_matches_with_provenance(self):
        captured_grounding = []
        captured_sql = []

        def real_sql_service(
            db_path,
            question,
            program,
            _sql_model,
            _answer_model,
            *,
            conversation_context=None,
            grounding_callable=None,
            **kwargs,
        ):
            def capture_grounding(current_question):
                grounded = grounding_callable(current_question)
                captured_grounding.append(grounded)
                return grounded

            result = run_ask_sql(
                db_path,
                question,
                program,
                lambda _prompt: (
                    "SELECT DISTINCT course_code, program "
                    "FROM v_plan_courses WHERE program = 'DSBA' "
                    "ORDER BY course_code"
                ),
                lambda _prompt: self.fail("canonical grounding must answer"),
                conversation_context=conversation_context,
                grounding_callable=capture_grounding,
                **kwargs,
            )
            captured_sql.append(result)
            return result

        self.sql_service.side_effect = real_sql_service
        with patch.object(main, "answer_hard_question", return_value=None):
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "วิชาไหนเรียนเกี่ยวกับ data",
                    "conversation_context": {
                        "program": "DSBA",
                        "catalog_key": "dsba-2565",
                    },
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "answer", payload)
        self.assertTrue(payload["provenance"])
        self.assertEqual(len(captured_grounding), 1)
        grounded = captured_grounding[0]
        qa_result = grounded.get("result")
        self.assertEqual(grounded.get("status"), "answer")
        self.assertTrue(hasattr(qa_result, "claims"))
        topic_claims = [claim for claim in qa_result.claims if claim.operation == "list"]
        self.assertTrue(topic_claims)
        self.assertTrue(all(claim.status == "complete" for claim in topic_claims))
        self.assertTrue(
            all(claim.effective_scope.catalog_key == "dsba-2565" for claim in topic_claims)
        )
        self.assertTrue(all(claim.provenance for claim in topic_claims))
        matched_courses = [
            course for claim in topic_claims for course in claim.value
        ]
        matched_codes = {course["course_code"] for course in matched_courses}
        sql_codes = {row["course_code"] for row in captured_sql[0]["rows"]}
        self.assertTrue(matched_codes)
        self.assertLess(len(matched_codes), len(sql_codes))
        self.assertTrue(matched_codes <= sql_codes)
        self.assertTrue(
            all(course.get("description_evidence") for course in matched_courses)
        )

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

    def test_easy_medium_wording_matrix_uses_real_sql_and_grounding_path(self):
        """Exercise API -> ask_sql -> grounding -> canonical QA for user wording."""
        cases = (
            ("A", "วิชาที่สอนในปี 1 เทอม 1", ("list",)),
            ("A-variant", "รายวิชาในปี 1 เทอม 1 มีอะไรบ้าง", ("list",)),
            ("B", "ปี 1 เทอม 1 มีวิชาอะไรบ้าง", ("list",)),
            (
                "C",
                "วิชาใดมีวิชาบังคับก่อนบ้าง บอกชื่อและรหัสวิชามา",
                ("prerequisite",),
            ),
            ("D", "วิชาเลือกมีอะไรบ้าง", ("list",)),
            ("D-variant", "ขอรายชื่อวิชาเลือก", ("list",)),
            ("E", "06026212 ต้องเรียนอะไรมาก่อน", ("describe", "prerequisite")),
            ("F", "06026212 กี่หน่วยกิต", ("sum_credits",)),
            ("G", "06026212 เรียนปีไหน", ("placement",)),
            ("H", "ปี 3 เทอม 1 รวมกี่หน่วยกิต", ("sum_credits",)),
            ("I", "06026212 เรียนเกี่ยวกับอะไร", ("describe",)),
        )
        diagnostics = []

        def deterministic_model(prompt, **_options):
            intent_prompt_prefix = (
                "ROLE: You interpret Thai university curriculum questions"
            )
            if prompt.startswith(intent_prompt_prefix):
                diagnostics[-1]["intent_calls"] += 1
                return json.dumps(
                    {
                        "intent": "course_list_query",
                        "proposed_program": None,
                        "proposed_plans": [],
                        "proposed_years": [],
                        "proposed_semesters": [],
                        "course_codes": [],
                        "topic": None,
                        "requested_facts": ["course_list"],
                        "judgement_dimension": None,
                        "unresolved": [],
                    },
                    ensure_ascii=False,
                )
            diagnostics[-1]["sql_model_calls"] += 1
            return (
                "SELECT course_code, program, plan_key, year, semester "
                "FROM v_plan_courses"
            )

        def real_sql_service(
            db_path,
            question,
            program,
            _sql_model,
            _answer_model,
            *,
            conversation_context=None,
            grounding_callable=None,
            **kwargs,
        ):
            diag = diagnostics[-1]

            def capture_grounding(current_question):
                grounded = grounding_callable(current_question)
                diag["grounding_status"] = grounded.get("status")
                return grounded

            result = run_ask_sql(
                db_path,
                question,
                program,
                _sql_model,
                lambda _prompt: (_ for _ in ()).throw(
                    AssertionError("free-form SQL answer synthesis must not run")
                ),
                conversation_context=conversation_context,
                grounding_callable=capture_grounding,
                **kwargs,
            )
            diag["sql_rows"] = len(result.get("rows", ()))
            return result

        original_plan_evidence = rag_qa.plan_evidence

        def capture_plan_evidence(spec, *args, **kwargs):
            diagnostics[-1]["effective_operations"].append(list(spec.operations))
            diagnostics[-1]["planned_catalog_keys"].append(kwargs.get("catalog_key"))
            return original_plan_evidence(spec, *args, **kwargs)

        with (
            patch.object(main, "answer_hard_question", return_value=None),
            patch.object(main, "_lazy_provider", side_effect=deterministic_model),
            patch.object(main, "ask_sql", side_effect=real_sql_service),
            patch.object(rag_qa, "plan_evidence", side_effect=capture_plan_evidence),
        ):
            for label, question, expected_operations in cases:
                pre_context_spec = parse_query_spec(question)
                diag = {
                    "case": label,
                    "question": question,
                    "surface_operations": list(detect_surface_operations(question)),
                    "pre_context_operations": list(pre_context_spec.operations),
                    "effective_operations": [],
                    "planned_catalog_keys": [],
                    "sql_rows": 0,
                    "grounding_status": None,
                    "sql_model_calls": 0,
                    "intent_calls": 0,
                    "api_status": None,
                }
                diagnostics.append(diag)
                with self.subTest(case=label):
                    response = self.client.post(
                        "/api/ask",
                        json={
                            "question": question,
                            "conversation_context": {
                                "program": "DSBA",
                                "catalog_key": "dsba-2565",
                            },
                        },
                    )
                    diag["api_status"] = response.status_code
                    payload = response.json()
                    self.assertEqual(response.status_code, 200, payload)
                    self.assertEqual(payload["status"], "answer", payload)
                    self.assertEqual(payload["route"], "llm_sql")
                    self.assertTrue(payload["provenance"])
                    self.assertEqual(payload["next_context"]["catalog_key"], "dsba-2565")
                    self.assertTrue(
                        all(
                            reference.get("program") == "DSBA"
                            for reference in payload["provenance"]
                        )
                    )
                    self.assertTrue(diag["sql_rows"] > 0, diag)
                    self.assertEqual(diag["sql_model_calls"], 1, diag)
                    self.assertEqual(diag["grounding_status"], "answer", diag)
                    self.assertTrue(
                        any(
                            ops == list(expected_operations)
                            for ops in diag["effective_operations"]
                        ),
                        diag,
                    )
                    self.assertTrue(
                        all(key == "dsba-2565" for key in diag["planned_catalog_keys"]),
                        diag,
                    )
                    self.assertNotIn("clarify_catalog", payload["status"])
                    self.assertNotIn("insufficient_evidence", payload["status"])
                    if label == "D":
                        self.assertEqual(diag["pre_context_operations"], [], diag)
                    if label == "C":
                        for course_code in (
                            "06026201", "06026212", "06026213", "06026215", "06066102"
                        ):
                            self.assertIn(course_code, payload["answer"])
                    elif label == "D-variant":
                        self.assertEqual(diag["intent_calls"], 1, diag)
                    elif label == "E":
                        self.assertIn("06066300", payload["answer"])
                    elif label == "F":
                        self.assertIn("3 หน่วยกิต", payload["answer"])
                    elif label == "G":
                        self.assertIn("ปี 3", payload["answer"])
                    elif label == "H":
                        self.assertIn("15 หน่วยกิต", payload["answer"])
                    elif label == "I":
                        self.assertIn("DATA WAREHOUSING", payload["answer"])

        print("EASY_MEDIUM_E2E_DIAGNOSTICS=" + json.dumps(diagnostics, ensure_ascii=False))

    def test_thai_exact_course_credit_overrides_previous_term_aggregation(self):
        exact_question = "แคลคูลัส 2 มีหน่วยกิตเท่าไหร่"
        parsed_exact = parse_query_spec(
            exact_question,
            has_validated_context_scope=True,
        )
        self.assertEqual(parsed_exact.course_name, "แคลคูลัส 2")
        self.assertEqual(parsed_exact.operations, ("sum_credits",))

        def deterministic_sql_provider(_prompt, **_options):
            return (
                "SELECT course_code, program, plan_key, year, semester "
                "FROM v_plan_courses"
            )

        exact_contexts = []

        def real_sql_service(
            db_path,
            question,
            program,
            sql_model,
            answer_model,
            *,
            conversation_context=None,
            **kwargs,
        ):
            if question == exact_question:
                exact_contexts.append(conversation_context)
            return run_ask_sql(
                db_path,
                question,
                program,
                sql_model,
                answer_model,
                conversation_context=conversation_context,
                **kwargs,
            )

        initial_context = {"program": "DSBA", "catalog_key": "dsba-2565"}
        broad_question = (
            "DSBA no_coop ปี 1 ปี 2 ปี 3 ปี 4 เทอม 1 เทอม 2 "
            "แต่ละเทอมรวมกี่หน่วยกิต"
        )
        with (
            patch.object(main, "ask_sql", side_effect=real_sql_service),
            patch.object(main, "answer_hard_question", return_value=None),
            patch.object(main, "_lazy_provider", side_effect=deterministic_sql_provider),
        ):
            previous = self.client.post(
                "/api/ask",
                json={"question": broad_question, "conversation_context": initial_context},
            )
            self.assertEqual(previous.status_code, 200, previous.json())
            previous_context = previous.json()["next_context"]
            self.assertEqual(previous_context["plan"], "no_coop")
            self.assertEqual(previous_context["years"], [1, 2, 3, 4])
            self.assertEqual(previous_context["semesters"], [1, 2])
            self.assertEqual(previous_context["operations"], ["sum_credits"])

            fresh = self.client.post(
                "/api/ask",
                json={"question": exact_question, "conversation_context": initial_context},
            )
            same_session = self.client.post(
                "/api/ask",
                json={"question": exact_question, "conversation_context": previous_context},
            )

        for response in (fresh, same_session):
            self.assertEqual(response.status_code, 200, response.json())
            payload = response.json()
            self.assertEqual(payload["status"], "answer", payload)
            self.assertEqual(payload["route"], "llm_sql")
            self.assertIn("06026201", payload["answer"])
            self.assertIn("3 หน่วยกิต", payload["answer"])
            self.assertNotIn("ลงทะเบียนรวม", payload["answer"])
            self.assertLessEqual(payload["answer"].count("06026201"), 2)
            self.assertTrue(payload["provenance"])
            next_context = payload["next_context"]
            self.assertEqual(next_context["catalog_key"], "dsba-2565")
            self.assertEqual(next_context["course_code"], "06026201")
            self.assertNotIn("years", next_context)
            self.assertNotIn("semesters", next_context)
            self.assertNotIn("plan", next_context)
        self.assertEqual(
            exact_contexts,
            [
                {"program": "DSBA", "catalog_key": "dsba-2565"},
                {"program": "DSBA", "catalog_key": "dsba-2565"},
            ],
        )

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

    def test_hard_seven_term_without_plan_returns_plan_clarification(self):
        intent = {
            "task_type": "seven_term_plan",
            "program": "DSBA",
            "plan": None,
            "left_plan": None,
            "right_plan": None,
            "target_course_code": None,
            "horizon_terms": 7,
        }
        with patch.object(main, "_lazy_provider", return_value=json.dumps(intent)) as provider:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "ถ้าอยากเรียนจบใน 3.5 ปีต้องทำยังไง",
                    "conversation_context": {
                        "program": "DSBA",
                        "catalog_key": "dsba-2565",
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["route"], "hard")
        self.assertEqual(payload["hard_task_type"], "seven_term_plan")
        self.assertEqual(payload["status"], "clarification_required")
        self.assertEqual(payload["action"], "plan_required")
        self.assertEqual(payload["provenance"], [])
        self.assertNotIn("plan_results", payload)
        self.assertIn("เลือกแผนหลักสูตร", payload["answer"])
        self.assertNotIn("ผลแผน coop", payload["answer"])
        self.assertNotIn("ผลแผน no_coop", payload["answer"])
        self.assertEqual(self.sql_service.call_count, 0)
        provider.assert_called_once()

    def test_h4_plan_selection_then_term_continuation_stays_plan_scoped(self):
        plan_intent = {
            "task_type": "seven_term_plan",
            "program": "DSBA",
            "plan": None,
            "left_plan": None,
            "right_plan": None,
            "target_course_code": None,
            "horizon_terms": 7,
        }
        with patch.object(main, "_lazy_provider", return_value=json.dumps(plan_intent)):
            pending = self.client.post(
                "/api/ask",
                json={
                    "question": "อยากจบใน 3.5 ปีต้องทำยังไง",
                    "conversation_context": {
                        "program": "DSBA",
                        "catalog_key": "dsba-2565",
                    },
                },
            ).json()
        self.assertEqual(pending["status"], "clarification_required")
        self.assertEqual(pending["action"], "plan_required")

        for plan, other_plan in (("coop", "no_coop"), ("no_coop", "coop")):
            with self.subTest(plan=plan):
                coop_intent = dict(plan_intent, plan=plan)
                with patch.object(main, "_lazy_provider", return_value=json.dumps(coop_intent)):
                    scoped = self.client.post(
                        "/api/ask",
                        json={
                            "question": "อยากจบใน 3.5 ปีต้องทำยังไง",
                            "conversation_context": {
                                "program": "DSBA",
                                "catalog_key": "dsba-2565",
                                "plan": plan,
                            },
                        },
                    ).json()
                self.assertEqual(scoped["hard_task_type"], "seven_term_plan")
                self.assertNotEqual(scoped["status"], "clarification_required")
                self.assertTrue(scoped["provenance"])
                self.assertNotIn("plan_results", scoped)
                self.assertNotIn(f"ผลแผน {other_plan}", scoped["answer"])
                self.assertEqual(scoped["next_context"]["study_plan_context"]["plan"], plan)
                context = scoped["next_context"]

                credit = self.client.post(
                    "/api/ask",
                    json={"question": "ปี 1 เทอม 1 หน่วยกิตเท่าไหร่", "conversation_context": context},
                ).json()
                self.assertEqual(credit["status"], "answer")
                self.assertEqual(credit["hard_task_type"], "seven_term_plan")
                self.assertTrue(credit["provenance"])
                self.assertLess(len(credit["provenance"]), len(scoped["provenance"]))

                partial = self.client.post(
                    "/api/ask",
                    json={"question": "ปี 3 เทอม 1 ล่ะ", "conversation_context": context},
                ).json()
                self.assertEqual(partial["status"], "answer")
                self.assertIn("ยืนยันได้อย่างน้อย", partial["answer"])
                self.assertTrue(partial["provenance"])
                self.assertLess(len(partial["provenance"]), len(scoped["provenance"]))

                course_list = self.client.post(
                    "/api/ask",
                    json={"question": "วิชาในปี 3 เทอม 1 มีอะไรบ้าง", "conversation_context": context},
                ).json()
                self.assertEqual(course_list["status"], "answer")
                self.assertTrue(course_list["provenance"])
                self.assertLess(len(course_list["provenance"]), len(scoped["provenance"]))

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

    # --- Micro-task 1: emitted-context round-trip contract ---
    def _ask_with_stub_provider(self, question, context):
        """Run POST /api/ask with the real ask_sql but a stub model (no network)."""

        def stub_model(prompt, **options):
            if prompt.startswith(
                "ROLE: You interpret Thai university curriculum questions"
            ):
                return json.dumps(
                    {
                        "intent": "course_list_query",
                        "proposed_program": None,
                        "proposed_plans": [],
                        "proposed_years": [],
                        "proposed_semesters": [],
                        "course_codes": [],
                        "topic": None,
                        "requested_facts": ["course_list"],
                        "judgement_dimension": None,
                        "unresolved": [],
                    },
                    ensure_ascii=False,
                )
            if "Matching database rows exist" in prompt:
                return "พบข้อมูลรายวิชาที่ตรงกับคำถาม"
            return (
                "SELECT course_code, program, plan_key, year, semester "
                "FROM v_plan_courses"
            )

        with (
            patch.object(main, "answer_hard_question", return_value=None),
            patch.object(main, "_lazy_provider", side_effect=stub_model),
            patch.object(
                main, "ask_sql", side_effect=lambda *args, **kwargs: run_ask_sql(*args, **kwargs)
            ),
        ):
            response = self.client.post(
                "/api/ask",
                json={"question": question, "conversation_context": context},
            )
        self.assertEqual(response.status_code, 200, response.json())
        return response.json()

    def test_grounded_course_next_context_round_trips_without_invalid_context(self):
        first = self._ask_with_stub_provider(
            "06016414 คือวิชาอะไร",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(first["status"], "answer")
        self.assertTrue(first["provenance"])
        emitted = first["next_context"]
        self.assertIsInstance(emitted, dict)

        second = self._ask_with_stub_provider("มันกี่หน่วย", emitted)
        self.assertNotEqual(second["status"], "error")
        self.assertNotEqual(second.get("action"), "invalid_context")

    def test_emitted_course_code_context_is_valid_next_request_input(self):
        # Exact writer shape observed from the grounded exact-course path.
        emitted = {
            "program": "IT",
            "catalog_key": "it-2565",
            "course_code": "06016414",
            "operations": ["identity"],
        }
        payload = self._ask_with_stub_provider("มันกี่หน่วย", emitted)
        self.assertNotEqual(payload["status"], "error")
        self.assertNotEqual(payload.get("action"), "invalid_context")

    def test_explicit_new_course_overrides_stale_course_context(self):
        first = self._ask_with_stub_provider(
            "06016414 คือวิชาอะไร",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(first["status"], "answer")

        second = self._ask_with_stub_provider(
            "06016438 กี่หน่วย", first["next_context"]
        )
        self.assertNotEqual(second["status"], "error")
        self.assertNotEqual(second.get("action"), "invalid_context")
        self.assertEqual(second["status"], "answer")
        self.assertIn("06016438", second["answer"])

    def test_fresh_semantic_collection_replaces_exact_placement_context(self):
        scope = {"program": "IT", "catalog_key": "it-2565"}
        fresh = self._ask_with_stub_provider(
            "มีวิชาเกี่ยวกับ cyber security อะไรบ้างครับ", scope
        )
        self.assertEqual(fresh["status"], "answer")
        self.assertIn("06016438", fresh["answer"])
        self.assertIn("06016464", fresh["answer"])

        first = self._ask_with_stub_provider("06016414 เรียนตอนไหน", scope)
        self.assertEqual(first["status"], "answer")
        second = self._ask_with_stub_provider(
            "มีวิชาเกี่ยวกับ cyber security อะไรบ้างครับ",
            first["next_context"],
        )

        self.assertEqual(second["status"], "answer")
        self.assertIn("06016438", second["answer"])
        self.assertIn("06016464", second["answer"])

    def test_fresh_semantic_collection_replaces_exact_credit_context(self):
        first = self._ask_with_stub_provider(
            "06016414 กี่หน่วยกิต", {"program": "IT", "catalog_key": "it-2565"}
        )
        self.assertEqual(first["status"], "answer")
        second = self._ask_with_stub_provider(
            "มีวิชาเกี่ยวกับ cyber security อะไรบ้าง", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        self.assertIn("06016438", second["answer"])
        self.assertIn("06016464", second["answer"])

    def test_semantic_topic_does_not_disable_true_course_pronoun_followup(self):
        first = self._ask_with_stub_provider(
            "06016414 เรียนตอนไหน", {"program": "IT", "catalog_key": "it-2565"}
        )
        second = self._ask_with_stub_provider("แล้วตัวนี้กี่หน่วย", first["next_context"])
        self.assertEqual(second["status"], "answer")
        self.assertIn("06016414", second["answer"])

    def test_fresh_semantic_query_preserves_current_year_not_stale_course_scope(self):
        prior = {
            "program": "IT",
            "catalog_key": "it-2565",
            "course_code": "06016414",
            "operations": ["placement"],
            "years": [2],
            "semesters": [2],
        }
        payload = self._ask_with_stub_provider(
            "ปี 3 มีวิชาเกี่ยวกับ cyber security อะไรบ้าง", prior
        )
        self.assertEqual(payload["status"], "answer")
        self.assertEqual(payload["next_context"].get("years"), [3])
        self.assertNotIn("06016414", json.dumps(payload["next_context"]))

        semester_payload = self._ask_with_stub_provider(
            "เทอม 2 มีวิชาเกี่ยวกับ cyber security อะไรบ้าง", prior
        )
        self.assertEqual(semester_payload["status"], "answer")
        self.assertEqual(semester_payload["next_context"].get("semesters"), [2])
        self.assertNotIn("06016414", json.dumps(semester_payload["next_context"]))

    def test_fresh_semantic_query_preserves_validated_plan(self):
        prior = {
            "program": "IT",
            "catalog_key": "it-2565",
            "plan": "coop",
            "course_code": "06016414",
            "operations": ["placement"],
        }
        payload = self._ask_with_stub_provider(
            "มีวิชาเกี่ยวกับ cyber security อะไรบ้าง", prior
        )
        self.assertEqual(payload["status"], "answer")
        self.assertEqual(payload["next_context"].get("plan"), "coop")
        self.assertNotIn("06016414", json.dumps(payload["next_context"]))

    def test_new_semantic_result_set_replaces_prior_set_for_ordinal(self):
        scope = {"program": "IT", "catalog_key": "it-2565"}
        first = self._ask_with_stub_provider(
            "มีวิชาเกี่ยวกับ database อะไรบ้าง", scope
        )
        self.assertEqual(first["status"], "answer")
        topic_a_codes = [
            item["course_code"] for item in first["next_context"].get("result_courses", [])
        ]

        second = self._ask_with_stub_provider(
            "มีวิชาเกี่ยวกับ cyber security อะไรบ้าง", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        topic_b_codes = [
            item["course_code"] for item in second["next_context"].get("result_courses", [])
        ]
        self.assertTrue(topic_b_codes)
        self.assertNotEqual(topic_a_codes, topic_b_codes)

        third = self._ask_with_stub_provider("ตัวแรกกี่หน่วย", second["next_context"])
        self.assertEqual(third["status"], "answer")
        self.assertIn(topic_b_codes[0], third["answer"])

    def test_unknown_context_key_remains_rejected(self):
        response = self.client.post(
            "/api/ask",
            json={
                "question": "06016414 คือวิชาอะไร",
                "conversation_context": {
                    "program": "IT",
                    "catalog_key": "it-2565",
                    "injected_fact": "06016414",
                },
            },
        )
        self.assertEqual(response.status_code, 422)

    # --- Micro-task 2: targetless detail/reference fail-closed invariant ---
    def _assert_targetless_fail_closed(self, question):
        payload = self._ask_with_stub_provider(
            question, {"program": "IT", "catalog_key": "it-2565"}
        )
        self.assertNotEqual(payload["status"], "error")
        self.assertNotEqual(payload.get("action"), "invalid_context")
        self.assertEqual(payload["status"], "insufficient_evidence")
        self.assertEqual(payload["provenance"], [])
        self.assertNotRegex(payload["answer"], r"0\d{7}")
        return payload

    def test_targetless_credits_fail_closed_without_dump(self):
        self._assert_targetless_fail_closed("กี่หน่วยอะ")

    def test_targetless_placement_this_fails_closed_without_dump(self):
        self._assert_targetless_fail_closed("ตัวนี้เรียนตอนไหน")

    def test_targetless_placement_that_fails_closed_without_dump(self):
        self._assert_targetless_fail_closed("ตัวนั้นเรียนตอนไหน")

    def test_resolved_course_followup_still_answers(self):
        first = self._ask_with_stub_provider(
            "06016414 คือวิชาอะไร",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(first["status"], "answer")

        second = self._ask_with_stub_provider("มันกี่หน่วย", first["next_context"])
        self.assertNotEqual(second["status"], "error")
        self.assertNotEqual(second.get("action"), "invalid_context")
        self.assertEqual(second["status"], "answer")
        self.assertIn("06016414", second["answer"])

    def test_scoped_semester_aggregate_stays_working(self):
        payload = self._ask_with_stub_provider(
            "IT ปี1เทอม1 รวมกี่หน่วย",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(payload["status"], "answer")
        self.assertTrue(payload["provenance"])
        self.assertEqual(payload["next_context"]["years"], [1])
        self.assertEqual(payload["next_context"]["semesters"], [1])

    def test_program_total_from_requirements_stays_working(self):
        payload = self._ask_with_stub_provider(
            "IT แผนสหกิจรวมทั้งหมดกี่หน่วยกิต",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(payload["status"], "answer")
        self.assertIn("129", payload["answer"])

    def test_scoped_semester_list_stays_working(self):
        payload = self._ask_with_stub_provider(
            "IT ปี1เทอม1 เรียนอะไรบ้าง",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(payload["status"], "answer")
        self.assertTrue(payload["provenance"])
        self.assertIn("06016401", payload["answer"])

    # --- Micro-task 3: NULL placement must not become concrete timing ---
    def _assert_null_placement_answer(self, payload):
        self.assertEqual(payload["status"], "answer")
        self.assertTrue(payload["provenance"])
        self.assertIn(
            "ไม่มีข้อมูลปี/ภาคเรียนที่แน่นอนในหลักสูตร", payload["answer"]
        )
        self.assertNotIn("เรียนในปี", payload["answer"])
        self.assertNotIn("สามารถเลือกจัดเรียนได้ใน", payload["answer"])
        return payload

    def test_null_placement_course_states_absence_without_fabrication(self):
        payload = self._ask_with_stub_provider(
            "06016438 เรียนตอนไหน",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self._assert_null_placement_answer(payload)
        self.assertIn("06016438", payload["answer"])

    def test_explicit_plan_scope_does_not_manufacture_placement(self):
        for question, plan_label in (
            ("IT สหกิจ 06016438 เรียนตอนไหน", "สหกิจ"),
            ("IT ไม่สหกิจ 06016438 เรียนตอนไหน", "ไม่สหกิจ"),
        ):
            with self.subTest(question=question):
                payload = self._ask_with_stub_provider(
                    question, {"program": "IT", "catalog_key": "it-2565"}
                )
                self._assert_null_placement_answer(payload)
                self.assertIn(plan_label, payload["answer"])

    def test_fixed_placement_course_still_returns_concrete_timing(self):
        payload = self._ask_with_stub_provider(
            "06016414 เรียนตอนไหน",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(payload["status"], "answer")
        self.assertTrue(payload["provenance"])
        self.assertIn("เรียนในปี 2 ภาคเรียนที่ 2", payload["answer"])

    def test_targetless_placement_still_fail_closed(self):
        payload = self._ask_with_stub_provider(
            "ตัวนี้เรียนตอนไหน", {"program": "IT", "catalog_key": "it-2565"}
        )
        self.assertEqual(payload["status"], "insufficient_evidence")
        self.assertEqual(payload["provenance"], [])

    def test_placed_course_followup_still_returns_timing(self):
        first = self._ask_with_stub_provider(
            "06016414 คือวิชาอะไร",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(first["status"], "answer")

        second = self._ask_with_stub_provider(
            "มันเรียนตอนไหน", first["next_context"]
        )
        self.assertNotEqual(second["status"], "error")
        self.assertNotEqual(second.get("action"), "invalid_context")
        self.assertEqual(second["status"], "answer")
        self.assertIn("เรียนในปี 2 ภาคเรียนที่ 2", second["answer"])

    # --- Micro-task 4: safe failures preserve validated scope ---
    def test_scope_survives_safe_failure_across_semester_followup(self):
        first = self._ask_with_stub_provider(
            "IT ปี1เทอม1 เรียนอะไรบ้าง",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(first["status"], "answer")
        emitted = first["next_context"]
        self.assertEqual(emitted["years"], [1])
        self.assertEqual(emitted["semesters"], [1])

        # Simulate the exact safe-failure envelope the real pipeline returns
        # for an unanswerable follow-up (explicit next_context None).
        self.sql_service.return_value = {
            "status": "insufficient_evidence",
            "answer": "ไม่พบหลักฐานที่มีแหล่งอ้างอิงเพียงพอสำหรับคำตอบนี้",
            "provenance": [],
            "next_context": None,
        }
        with patch.object(main, "answer_hard_question", return_value=None):
            response = self.client.post(
                "/api/ask",
                json={"question": "รวมกี่หน่วย", "conversation_context": emitted},
            )
        self.assertEqual(response.status_code, 200)
        second = response.json()
        self.assertEqual(second["status"], "insufficient_evidence")
        preserved = second["next_context"]
        self.assertEqual(preserved["program"], "IT")
        self.assertEqual(preserved["catalog_key"], "it-2565")
        self.assertEqual(preserved["years"], [1])
        self.assertEqual(preserved["semesters"], [1])

        third = self._ask_with_stub_provider("แล้วเทอม 2 ล่ะ", preserved)
        self.assertNotEqual(third["status"], "error")
        self.assertNotEqual(third.get("action"), "invalid_context")
        self.assertEqual(third["next_context"]["years"], [1])
        self.assertEqual(third["next_context"]["semesters"], [2])

    def test_course_target_survives_unrelated_safe_failure(self):
        first = self._ask_with_stub_provider(
            "06016414 คือวิชาอะไร",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(first["status"], "answer")

        second = self._ask_with_stub_provider(
            "เทอมหน้าวิชานี้เปิดมั้ย", first["next_context"]
        )
        self.assertEqual(second["status"], "insufficient_evidence")
        focus = (second["next_context"] or {}).get("focus_course") or {}
        self.assertEqual(focus.get("course_code"), "06016414")

        third = self._ask_with_stub_provider("มันกี่หน่วย", second["next_context"])
        self.assertEqual(third["status"], "answer")
        self.assertIn("06016414", third["answer"])

    def test_explicit_course_overrides_preserved_stale_target(self):
        first = self._ask_with_stub_provider(
            "06016414 คือวิชาอะไร",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        second = self._ask_with_stub_provider(
            "เทอมหน้าวิชานี้เปิดมั้ย", first["next_context"]
        )
        self.assertEqual(second["status"], "insufficient_evidence")

        third = self._ask_with_stub_provider(
            "06016438 กี่หน่วย", second["next_context"]
        )
        self.assertNotEqual(third["status"], "error")
        self.assertNotEqual(third.get("action"), "invalid_context")
        self.assertIn("06016438", third["answer"])
        self.assertNotIn("06016414", json.dumps(third["next_context"] or {}))

    def test_targetless_fresh_session_gains_no_state(self):
        payload = self._ask_with_stub_provider(
            "กี่หน่วยอะ", {"program": "IT", "catalog_key": "it-2565"}
        )
        self.assertEqual(payload["status"], "insufficient_evidence")
        survived = payload["next_context"] or {}
        for key in ("course_code", "focus_course", "years", "semesters",
                    "result_courses", "operations"):
            self.assertNotIn(key, survived)

    def test_program_switch_still_clears_stale_course_state(self):
        first = self._ask_with_stub_provider(
            "06016414 คือวิชาอะไร",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        with patch.object(main, "answer_hard_question", return_value=None):
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "แล้ว DSBA สหกิจล่ะ",
                    "conversation_context": first["next_context"],
                },
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["action"], "catalog_required")
        pending = payload["next_context"] or {}
        self.assertEqual(pending.get("program"), "DSBA")
        self.assertNotIn("course_code", pending)
        self.assertNotIn("focus_course", pending)
        self.assertNotIn("06016414", json.dumps(pending))

    # --- Micro-task 7: explicit plan negation dominates positive tokens ---
    def test_plan_override_coop_to_no_coop_wins(self):
        first = self._ask_with_stub_provider(
            "IT สหกิจนี่รวมกี่หน่วยนะ",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(first["status"], "answer")
        self.assertEqual(first["next_context"].get("plan"), "coop")

        second = self._ask_with_stub_provider(
            "แล้วไม่สหกิจล่ะ", first["next_context"]
        )
        self.assertNotEqual(second["status"], "error")
        self.assertEqual(second["next_context"].get("plan"), "no_coop")
        self.assertIn("ไม่สหกิจ", second["answer"])

    def test_plan_override_no_coop_to_coop_wins(self):
        first = self._ask_with_stub_provider(
            "IT ไม่สหกิจ ปี 3 เทอม 2 รวมกี่หน่วย",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(first["status"], "answer")
        self.assertEqual(first["next_context"].get("plan"), "no_coop")

        second = self._ask_with_stub_provider(
            "แล้วสหกิจล่ะ", first["next_context"]
        )
        self.assertNotEqual(second["status"], "error")
        self.assertEqual(second["next_context"].get("plan"), "coop")

    def test_plan_sensitive_retrieval_uses_intended_canonical_partition(self):
        coop = self._ask_with_stub_provider(
            "IT สหกิจ ปี 3 เทอม 2 รวมกี่หน่วย",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        no_coop = self._ask_with_stub_provider(
            "IT ไม่สหกิจ ปี 3 เทอม 2 รวมกี่หน่วย",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        for payload in (coop, no_coop):
            self.assertEqual(payload["status"], "answer")
            self.assertTrue(payload["provenance"])
        self.assertIn("06016481", coop["answer"])
        self.assertNotIn("06066100", coop["answer"])
        self.assertIn("06066100", no_coop["answer"])
        self.assertNotIn("06016481", no_coop["answer"])

    def test_no_plan_signal_infers_no_plan(self):
        payload = self._ask_with_stub_provider(
            "IT ปี1เทอม1 เรียนไรบ้างอะ",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(payload["status"], "answer")
        self.assertNotIn("plan", payload["next_context"] or {})

    def test_unknown_key_rejected_before_preservation(self):
        response = self.client.post(
            "/api/ask",
            json={
                "question": "กี่หน่วยอะ",
                "conversation_context": {
                    "program": "IT",
                    "catalog_key": "it-2565",
                    "injected_fact": "06016414",
                },
            },
        )
        self.assertEqual(response.status_code, 422)

    # --- Micro-task 8: ordinal references over retained canonical results ---
    def _cyber_search(self):
        return self._ask_with_stub_provider(
            "มีวิชาเกี่ยวกับ cybersecurity อะไรบ้าง",
            {"program": "IT", "catalog_key": "it-2565"},
        )

    def _assert_retained_identities_only(self, next_context):
        self.assertIsInstance(next_context, dict)
        courses = next_context.get("result_courses")
        self.assertIsInstance(courses, list)
        self.assertTrue(courses)
        for course in courses:
            self.assertIsInstance(course, dict)
            self.assertLessEqual(
                set(course),
                {"program", "course_code", "catalog_key", "course_name"},
            )
            self.assertTrue(course.get("course_code"))
            self.assertTrue(course.get("program"))
        for key in (
            "credits",
            "credit_units",
            "prerequisite",
            "prerequisites",
            "placement",
            "description",
            "answer",
            "rows",
            "sql",
        ):
            self.assertNotIn(key, next_context)
            for course in courses:
                self.assertNotIn(key, course)

    def test_semantic_result_set_retained_as_canonical_identities(self):
        payload = self._cyber_search()
        self.assertEqual(payload["status"], "answer")
        self.assertTrue(payload["provenance"])
        retained = payload["next_context"] or {}
        self.assertEqual(
            [course["course_code"] for course in retained["result_courses"]],
            ["06016405", "06016438", "06016464"],
        )
        self.assertEqual(retained.get("result_scope_program"), "IT")
        self._assert_retained_identities_only(retained)

    def test_first_result_credit_followup_regrounded(self):
        first = self._cyber_search()
        second = self._ask_with_stub_provider(
            "ตัวแรกกี่หน่วย", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        self.assertIn("06016405", second["answer"])
        self.assertNotIn("06016438", second["answer"])
        self.assertTrue(second["provenance"])

    def test_three_semantic_ordinals_reground_and_fourth_fails_closed(self):
        first = self._cyber_search()
        for ordinal, code in ((1, "06016405"), (2, "06016438"), (3, "06016464")):
            with self.subTest(ordinal=ordinal):
                answer = self._ask_with_stub_provider(
                    f"ตัวที่ {ordinal}กี่หน่วย", first["next_context"]
                )
                self.assertEqual(answer["status"], "answer")
                self.assertIn(code, answer["answer"])
                self.assertTrue(answer["provenance"])
        invalid = self._ask_with_stub_provider("ตัวที่ 4กี่หน่วย", first["next_context"])
        self.assertEqual(invalid["status"], "insufficient_evidence")
        self.assertEqual(invalid["provenance"], [])

    def test_sequential_semantic_ordinals_keep_parent_set_and_explicit_override_clears_it(self):
        current = self._ask_with_stub_provider(
            "มีวิชาเกี่ยวกับ cyber security อะไรบ้างครับ",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        original_context = current["next_context"]
        codes = ["06016405", "06016438", "06016464"]
        for question, code in (
            ("ตัวแรกกี่หน่วยกิต", codes[0]),
            ("ตัวที่สองกี่หน่วยกิต", codes[1]),
            ("ตัวที่สามกี่หน่วยกิต", codes[2]),
        ):
            current = self._ask_with_stub_provider(question, current["next_context"])
            self.assertEqual(current["status"], "answer", question)
            self.assertIn(code, current["answer"])
            self.assertTrue(current["provenance"])
            context = current["next_context"]
            self.assertEqual([item["course_code"] for item in context["result_courses"]], codes)
            self.assertEqual(context["result_scope_program"], "IT")
            self.assertEqual(context["semantic_topic"], "cyber security")
            self.assertEqual(context["last_answer"]["course_code"], code)

        placement = self._ask_with_stub_provider("แล้วตัวที่สามเรียนช่วงไหน", current["next_context"])
        self.assertIn(placement["status"], ("answer", "insufficient_evidence"))
        self.assertNotIn("เรียนในปี", placement["answer"])
        invalid = self._ask_with_stub_provider("ตัวที่สี่กี่หน่วย", placement["next_context"])
        self.assertEqual(invalid["status"], "insufficient_evidence")
        self.assertEqual(invalid["provenance"], [])

        explicit = self._ask_with_stub_provider("06016414 กี่หน่วยกิต", original_context)
        self.assertEqual(explicit["status"], "answer")
        self.assertIn("06016414", explicit["answer"])
        self.assertNotIn("result_courses", explicit["next_context"])
        stale = self._ask_with_stub_provider("ตัวที่สองกี่หน่วย", explicit["next_context"])
        self.assertEqual(stale["status"], "insufficient_evidence")
        self.assertEqual(stale["provenance"], [])

    def test_semantic_heading_separates_search_scope_from_placement(self):
        answer = self._cyber_search()
        self.assertNotIn("ปี 1/2/3/4", answer["answer"])
        self.assertIn("พบรายวิชาที่เกี่ยวข้อง", answer["answer"])
        scoped = self._ask_with_stub_provider(
            "ปี 3 มีวิชาเกี่ยวกับ cloud อะไรบ้าง",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(scoped["status"], "answer")
        self.assertIn("ขอบเขตการค้นหา", scoped["answer"])
        self.assertIn("ปี 3", scoped["answer"])

    def test_second_result_placement_followup_without_fabrication(self):
        first = self._cyber_search()
        second = self._ask_with_stub_provider(
            "ตัวที่สองเรียนตอนไหน", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        self.assertIn("06016438", second["answer"])
        self.assertIn(
            "ไม่มีข้อมูลปี/ภาคเรียนที่แน่นอนในหลักสูตร", second["answer"]
        )
        self.assertNotIn("เรียนในปี", second["answer"])
        self.assertTrue(second["provenance"])

    def test_explicit_course_overrides_ordinal_result_context(self):
        first = self._cyber_search()

        def credit_intent(prompt, **options):
            if prompt.startswith(
                "ROLE: You interpret Thai university curriculum questions"
            ):
                return json.dumps(
                    {
                        "intent": "course_credit_query",
                        "proposed_program": None,
                        "proposed_plans": [],
                        "proposed_years": [],
                        "proposed_semesters": [],
                        "course_codes": ["06016414"],
                        "topic": None,
                        "requested_facts": ["course_credit"],
                        "judgement_dimension": None,
                        "unresolved": [],
                    },
                    ensure_ascii=False,
                )
            if "Matching database rows exist" in prompt:
                return "พบข้อมูลรายวิชาที่ตรงกับคำถาม"
            return (
                "SELECT course_code, program, plan_key, year, semester "
                "FROM v_plan_courses"
            )

        with (
            patch.object(main, "answer_hard_question", return_value=None),
            patch.object(main, "_lazy_provider", side_effect=credit_intent),
            patch.object(
                main, "ask_sql", side_effect=lambda *args, **kwargs: run_ask_sql(*args, **kwargs)
            ),
        ):
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "06016414 กี่หน่วย",
                    "conversation_context": first["next_context"],
                },
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "answer")
        self.assertIn("06016414", payload["answer"])
        self.assertNotIn("06016405", payload["answer"])
        self.assertNotIn("06016438", payload["answer"])

    def test_new_result_set_replaces_old_set(self):
        first = self._cyber_search()
        second = self._ask_with_stub_provider(
            "มีวิชาเกี่ยวกับ machine learning อะไรบ้าง", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        retained = second["next_context"] or {}
        codes = [course["course_code"] for course in retained["result_courses"]]
        self.assertIn("06016460", codes)
        self.assertNotIn("06016405", codes)
        self.assertNotIn("06016438", codes)

        third = self._ask_with_stub_provider("ตัวแรกกี่หน่วย", second["next_context"])
        self.assertEqual(third["status"], "answer")
        self.assertIn("06016435", third["answer"])

    def test_fresh_ordinal_fails_closed(self):
        payload = self._ask_with_stub_provider(
            "ตัวแรกกี่หน่วย", {"program": "IT", "catalog_key": "it-2565"}
        )
        self.assertEqual(payload["status"], "insufficient_evidence")
        self.assertEqual(payload["provenance"], [])
        self.assertNotIn("result_courses", payload["next_context"] or {})

    def test_out_of_range_ordinal_fails_closed(self):
        first = self._cyber_search()
        second = self._ask_with_stub_provider(
            "ตัวที่ 5 กี่หน่วย", first["next_context"]
        )
        self.assertEqual(second["status"], "insufficient_evidence")
        self.assertEqual(second["provenance"], [])
        # The valid set itself survives the failed reference.
        kept = second["next_context"] or {}
        self.assertEqual(
            [course["course_code"] for course in kept.get("result_courses", [])],
            ["06016405", "06016438", "06016464"],
        )

    def test_empty_result_search_ordinal_fails_closed(self):
        first = self._ask_with_stub_provider(
            "มีวิชาเกี่ยวกับ flurbnax quantum banana อะไรบ้าง",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(first["status"], "insufficient_evidence")
        self.assertNotIn("result_courses", first["next_context"] or {})

        second = self._ask_with_stub_provider(
            "ตัวแรกกี่หน่วย", first["next_context"]
        )
        self.assertEqual(second["status"], "insufficient_evidence")
        self.assertEqual(second["provenance"], [])

    def test_safe_failure_preserves_result_set_for_ordinal(self):
        first = self._cyber_search()
        second = self._ask_with_stub_provider(
            "อธิบายหน่อยว่าวิชานี้เรียนเกี่ยวกับอะไร", first["next_context"]
        )
        self.assertEqual(second["status"], "insufficient_evidence")
        kept = second["next_context"] or {}
        self.assertEqual(
            [course["course_code"] for course in kept.get("result_courses", [])],
            ["06016405", "06016438", "06016464"],
        )

        third = self._ask_with_stub_provider(
            "ตัวที่สองเรียนตอนไหน", second["next_context"]
        )
        self.assertEqual(third["status"], "answer")
        self.assertIn("06016438", third["answer"])

    def test_stale_cross_scope_results_cannot_resolve_ordinal(self):
        first = self._cyber_search()
        stale = dict(first["next_context"] or {})
        stale["catalog_key"] = "it-2560"
        second = self._ask_with_stub_provider("ตัวแรกกี่หน่วย", stale)
        self.assertEqual(second["status"], "insufficient_evidence")
        kept = second["next_context"] or {}
        self.assertNotIn("result_courses", kept)
        self.assertNotIn("06016405", json.dumps(kept))

    def test_current_pronoun_resolves_from_focus_not_results(self):
        first = self._ask_with_stub_provider(
            "06016414 คือวิชาอะไร",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(first["status"], "answer")

        second = self._ask_with_stub_provider(
            "ตัวนั้นกี่หน่วย", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        self.assertIn("06016414", second["answer"])

        search = self._cyber_search()
        third = self._ask_with_stub_provider(
            "ตัวนั้นกี่หน่วย", search["next_context"]
        )
        self.assertEqual(third["status"], "insufficient_evidence")

    # --- Micro-task 12: unresolved ordinal must never fall back to focus ---
    def _focus_only_context(self, course_code="06016405"):
        return {
            "program": "IT",
            "catalog_key": "it-2565",
            "course_code": course_code,
            "operations": ["sum_credits"],
            "focus_course": {
                "course_code": course_code,
                "program": "IT",
                "catalog_key": "it-2565",
            },
        }

    def test_unresolved_ordinal_never_falls_back_to_focus(self):
        search = self._cyber_search()
        credit = self._ask_with_stub_provider(
            "ตัวแรกกี่หน่วย", search["next_context"]
        )
        self.assertEqual(credit["status"], "answer")
        self.assertIn("06016405", credit["answer"])
        self.assertEqual(
            credit["next_context"]["result_courses"],
            search["next_context"]["result_courses"],
        )
        third = self._ask_with_stub_provider(
            "ตัวที่ 4เรียนตอนไหน", credit["next_context"]
        )
        self.assertEqual(third["status"], "insufficient_evidence")
        self.assertEqual(third["provenance"], [])
        self.assertNotIn("06016405", third["answer"])

    def test_first_ordinal_with_only_focus_fails_closed(self):
        payload = self._ask_with_stub_provider(
            "ตัวแรกกี่หน่วย", self._focus_only_context()
        )
        self.assertEqual(payload["status"], "insufficient_evidence")
        self.assertEqual(payload["provenance"], [])
        self.assertNotIn("06016405", payload["answer"])

    def test_out_of_range_ordinal_with_focus_present_fails_closed(self):
        context = self._focus_only_context("06016414")
        context["result_courses"] = [
            {"program": "IT", "course_code": "06016405",
             "catalog_key": "it-2565"},
            {"program": "IT", "course_code": "06016438",
             "catalog_key": "it-2565"},
        ]
        context["result_scope_program"] = "IT"
        payload = self._ask_with_stub_provider("ตัวที่ 5 กี่หน่วย", context)
        self.assertEqual(payload["status"], "insufficient_evidence")
        self.assertEqual(payload["provenance"], [])
        self.assertNotIn("06016414", payload["answer"])

    def test_valid_second_ordinal_resolves_target_identity(self):
        search = self._cyber_search()
        payload = self._ask_with_stub_provider(
            "ตัวที่สองเรียนตอนไหน", search["next_context"]
        )
        self.assertEqual(payload["status"], "answer")
        self.assertIn("06016438", payload["answer"])
        self.assertTrue(payload["provenance"])

    def test_valid_first_ordinal_resolves_target_identity(self):
        search = self._cyber_search()
        payload = self._ask_with_stub_provider(
            "ตัวแรกกี่หน่วย", search["next_context"]
        )
        self.assertEqual(payload["status"], "answer")
        self.assertIn("06016405", payload["answer"])
        self.assertNotIn("06016438", payload["answer"])
        self.assertTrue(payload["provenance"])

    def test_explicit_course_overrides_stale_focus_and_results(self):
        context = self._focus_only_context("06016414")
        context["result_courses"] = [
            {"program": "IT", "course_code": "06016405",
             "catalog_key": "it-2565"},
        ]
        context["result_scope_program"] = "IT"
        payload = self._ask_with_stub_provider("06016438 กี่หน่วย", context)
        self.assertEqual(payload["status"], "answer")
        self.assertIn("06016438", payload["answer"])

    def test_focus_pronoun_continuation_still_works(self):
        first = self._ask_with_stub_provider(
            "06016414 คือวิชาอะไร",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(first["status"], "answer")
        second = self._ask_with_stub_provider(
            "มันกี่หน่วย", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        self.assertIn("06016414", second["answer"])

    def test_tua_nan_pronoun_keeps_focus_semantics(self):
        first = self._ask_with_stub_provider(
            "06016414 คือวิชาอะไร",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(first["status"], "answer")
        second = self._ask_with_stub_provider(
            "ตัวนั้นกี่หน่วย", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        self.assertIn("06016414", second["answer"])

    def test_fresh_ordinal_still_fails_closed(self):
        payload = self._ask_with_stub_provider(
            "ตัวที่สองเรียนตอนไหน",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(payload["status"], "insufficient_evidence")
        self.assertEqual(payload["provenance"], [])

    def test_unresolved_ordinal_preserves_focus_without_new_set(self):
        payload = self._ask_with_stub_provider(
            "ตัวที่สองเรียนตอนไหน", self._focus_only_context()
        )
        self.assertEqual(payload["status"], "insufficient_evidence")
        kept = payload["next_context"] or {}
        focus = kept.get("focus_course") or {}
        self.assertEqual(focus.get("course_code"), "06016405")
        self.assertNotIn("result_courses", kept)

    # --- Retention fallback: canonical validation when SQL rows disagree ---
    def _retained(self, answer, rows, program="IT", catalog="it-2565"):
        return _retained_identities_from_answer(
            answer, rows, program, catalog, "cybersecurity", DB_PATH
        )

    def _narrow_rows(self):
        return [{"program": "IT", "course_code": "06016405",
                 "catalog_key": "it-2565"}]

    def test_narrow_sql_rows_still_retain_both_answer_codes(self):
        retained = self._retained(
            "วิชา 06016405 และ 06016438 เกี่ยวกับ cybersecurity",
            self._narrow_rows(),
        )
        self.assertIsNotNone(retained)
        self.assertEqual(
            [course["course_code"] for course in retained["result_courses"]],
            ["06016405", "06016438"],
        )
        self._assert_retained_identities_only(retained)

    def test_disagreeing_sql_code_does_not_enter_retained_set(self):
        rows = self._narrow_rows() + [{"program": "IT",
                                       "course_code": "06016464",
                                       "catalog_key": "it-2565"}]
        retained = self._retained(
            "วิชา 06016405 และ 06016438 เกี่ยวกับ cybersecurity", rows
        )
        self.assertIsNotNone(retained)
        self.assertEqual(
            [course["course_code"] for course in retained["result_courses"]],
            ["06016405", "06016438"],
        )

    def test_single_named_answer_code_retains_nothing(self):
        retained = self._retained("วิชา 06016405 เกี่ยวกับ cybersecurity",
                                  self._narrow_rows())
        self.assertIsNone(retained)

    def test_hallucinated_answer_code_never_retained(self):
        retained = self._retained(
            "วิชา 06016405 และ 99999999 เกี่ยวกับ cybersecurity",
            self._narrow_rows(),
        )
        self.assertIsNone(retained)

    def test_cross_scope_answer_code_not_retained(self):
        retained = self._retained(
            "วิชา 06016405 และ 06026100 เกี่ยวกับ cybersecurity",
            self._narrow_rows(),
        )
        self.assertIsNone(retained)

    def test_retained_order_matches_answer_mention_order(self):
        retained = self._retained(
            "วิชา 06016438 และ 06016405 เกี่ยวกับ cybersecurity",
            self._narrow_rows(),
        )
        self.assertIsNotNone(retained)
        self.assertEqual(
            [course["course_code"] for course in retained["result_courses"]],
            ["06016438", "06016405"],
        )

    # --- Micro-task 9: previous-answer explanation/source follow-ups ---
    def _ask_with_explain_stub(self, question, context):
        def stub_model(prompt, **options):
            if prompt.startswith(
                "ROLE: You interpret Thai university curriculum questions"
            ):
                return json.dumps(
                    {
                        "intent": "course_list_query",
                        "proposed_program": None,
                        "proposed_plans": [],
                        "proposed_years": [],
                        "proposed_semesters": [],
                        "course_codes": [],
                        "topic": None,
                        "requested_facts": ["course_list"],
                        "judgement_dimension": None,
                        "unresolved": [],
                    },
                    ensure_ascii=False,
                )
            if "Matching database rows exist" in prompt:
                return "พบข้อมูลรายวิชาที่ตรงกับคำถาม"
            if prompt.startswith("EXPLAIN_GROUNDED_EVIDENCE"):
                return "ไม่พบข้อมูลที่ตรงกัน"
            return (
                "SELECT course_code, program, plan_key, year, semester "
                "FROM v_plan_courses"
            )

        with (
            patch.object(main, "answer_hard_question", return_value=None),
            patch.object(main, "_lazy_provider", side_effect=stub_model),
            patch.object(
                main, "ask_sql", side_effect=lambda *args, **kwargs: run_ask_sql(*args, **kwargs)
            ),
        ):
            response = self.client.post(
                "/api/ask",
                json={"question": question, "conversation_context": context},
            )
        self.assertEqual(response.status_code, 200, response.json())
        return response.json()

    def _probation_answer(self):
        return self._ask_with_explain_stub("gpa เท่าไหร่ถึงติดโปร", None)

    def _ask_with_credit_explain_stub(self, question, context):
        def stub_model(prompt, **options):
            if prompt.startswith(
                "ROLE: You interpret Thai university curriculum questions"
            ):
                return json.dumps(
                    {
                        "intent": "course_credit_query",
                        "proposed_program": None,
                        "proposed_plans": [],
                        "proposed_years": [],
                        "proposed_semesters": [],
                        "course_codes": ["06016414"],
                        "topic": None,
                        "requested_facts": ["course_credit"],
                        "judgement_dimension": None,
                        "unresolved": [],
                    },
                    ensure_ascii=False,
                )
            if "Matching database rows exist" in prompt:
                return "พบข้อมูลรายวิชาที่ตรงกับคำถาม"
            if prompt.startswith("EXPLAIN_GROUNDED_EVIDENCE"):
                return "ไม่พบข้อมูลที่ตรงกัน"
            return (
                "SELECT course_code, program, plan_key, year, semester "
                "FROM v_plan_courses"
            )

        with (
            patch.object(main, "answer_hard_question", return_value=None),
            patch.object(main, "_lazy_provider", side_effect=stub_model),
            patch.object(
                main, "ask_sql", side_effect=lambda *args, **kwargs: run_ask_sql(*args, **kwargs)
            ),
        ):
            response = self.client.post(
                "/api/ask",
                json={"question": question, "conversation_context": context},
            )
        self.assertEqual(response.status_code, 200, response.json())
        return response.json()

    def test_policy_answer_emits_bounded_reference(self):
        first = self._probation_answer()
        self.assertEqual(first["status"], "answer")
        self.assertTrue(first["provenance"])
        last = (first["next_context"] or {}).get("last_answer")
        self.assertIsInstance(last, dict)
        self.assertEqual(last.get("route"), "policy")
        self.assertEqual(last.get("policy_kind"), "probation_entry")
        self.assertLessEqual(len(last.get("evidence_ids", [])), 50)

    def test_policy_explanation_reuses_canonical_basis(self):
        first = self._probation_answer()
        second = self._ask_with_explain_stub(
            "ขยายความหน่อย", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        self.assertTrue(second["provenance"])
        self.assertEqual(second["provenance"], first["provenance"])
        self.assertIn("2", second["answer"])
        self.assertNotIn("06016414", second["answer"])
        kept = (second["next_context"] or {}).get("last_answer") or {}
        self.assertEqual(kept.get("policy_kind"), "probation_entry")

    def test_policy_source_names_canonical_rule(self):
        first = self._ask_with_explain_stub(
            "อุทธรณ์คำสั่งลงโทษต้องยื่นภายในกี่วัน", None
        )
        self.assertEqual(first["status"], "answer")
        second = self._ask_with_explain_stub(
            "อยู่ในข้อไหน", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        self.assertIn("43", second["answer"])
        self.assertTrue(second["provenance"])

    def test_policy_rationale_fails_closed(self):
        first = self._ask_with_explain_stub(
            "อุทธรณ์คำสั่งลงโทษต้องยื่นภายในกี่วัน", None
        )
        second = self._ask_with_explain_stub(
            "ทำไมถึงกำหนด 30 วัน", first["next_context"]
        )
        self.assertEqual(second["status"], "insufficient_evidence")
        self.assertEqual(second["provenance"], [])

    def test_fresh_explanation_fails_closed(self):
        payload = self._ask_with_explain_stub("ขยายความหน่อย", None)
        self.assertEqual(payload["status"], "insufficient_evidence")
        self.assertEqual(payload["provenance"], [])

    def test_fresh_source_fails_closed(self):
        payload = self._ask_with_explain_stub("อยู่ในข้อไหน", None)
        self.assertEqual(payload["status"], "insufficient_evidence")
        self.assertEqual(payload["provenance"], [])

    def test_newest_answer_wins_over_older_reference(self):
        policy = self._probation_answer()
        course = self._ask_with_credit_explain_stub(
            "06016414 กี่หน่วย", {"program": "IT", "catalog_key": "it-2565"}
        )
        self.assertEqual(course["status"], "answer")
        last = (course["next_context"] or {}).get("last_answer") or {}
        self.assertEqual(last.get("route"), "course")
        explained = self._ask_with_credit_explain_stub(
            "ขยายความหน่อย", course["next_context"]
        )
        self.assertEqual(explained["status"], "answer")
        self.assertIn("06016414", explained["answer"])
        self.assertNotIn("ภาคทัณฑ์", explained["answer"])
        kept = (explained["next_context"] or {}).get("last_answer") or {}
        self.assertEqual(kept.get("route"), "course")
        self.assertEqual(kept.get("course_code"), "06016414")

    def test_course_credit_explanation_regrounded(self):
        first = self._ask_with_credit_explain_stub(
            "06016414 กี่หน่วย", {"program": "IT", "catalog_key": "it-2565"}
        )
        self.assertEqual(first["status"], "answer")
        second = self._ask_with_credit_explain_stub(
            "ขยายความหน่อย", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        self.assertTrue(second["provenance"])
        self.assertIn("06016414", second["answer"])

    def test_ordinal_chain_explanation_uses_resolved_course(self):
        search = self._ask_with_credit_explain_stub(
            "มีวิชาเกี่ยวกับ cybersecurity อะไรบ้าง",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        credit = self._ask_with_credit_explain_stub(
            "ตัวแรกกี่หน่วย", search["next_context"]
        )
        self.assertEqual(credit["status"], "answer")
        explained = self._ask_with_credit_explain_stub(
            "ขยายความหน่อย", credit["next_context"]
        )
        self.assertEqual(explained["status"], "answer")
        self.assertIn("06016405", explained["answer"])
        self.assertNotIn("06016438", explained["answer"])

    def test_safe_failure_preserves_reference(self):
        first = self._probation_answer()
        failed = self._ask_with_explain_stub(
            "กี่หน่วยอะ", first["next_context"]
        )
        self.assertEqual(failed["status"], "insufficient_evidence")
        kept = (failed["next_context"] or {}).get("last_answer") or {}
        self.assertEqual(kept.get("policy_kind"), "probation_entry")
        explained = self._ask_with_explain_stub(
            "ขยายความหน่อย", failed["next_context"]
        )
        self.assertEqual(explained["status"], "answer")
        self.assertTrue(explained["provenance"])

    def test_cross_scope_reset_drops_reference(self):
        first = self._ask_with_credit_explain_stub(
            "06016414 กี่หน่วย", {"program": "IT", "catalog_key": "it-2565"}
        )
        stale = dict(first["next_context"] or {})
        stale["catalog_key"] = "it-2560"
        second = self._ask_with_explain_stub("ขยายความหน่อย", stale)
        self.assertEqual(second["status"], "insufficient_evidence")
        self.assertNotIn(
            "06016414", json.dumps(second["next_context"] or {})
        )

    def test_malformed_last_answer_rejected(self):
        for bad in (
            {"route": "policy", "policy_kind": "probation_entry",
             "evidence_ids": ["rule:1"] * 51},
            {"route": "policy", "policy_kind": "probation_entry",
             "evidence_ids": "rule:22"},
            {"route": "course", "course_code": "06016414",
             "operations": ["sum_credits"], "injected": True},
        ):
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "ขยายความหน่อย",
                    "conversation_context": {
                        "program": "IT",
                        "catalog_key": "it-2565",
                        "last_answer": bad,
                    },
                },
            )
            self.assertEqual(response.status_code, 422, bad)

    # --- Micro-task 11: audited follow-up alias families ---
    def test_new_explain_alias_regrounds_policy(self):
        first = self._probation_answer()
        second = self._ask_with_explain_stub(
            "ขยายอีกหน่อย", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        self.assertTrue(second["provenance"])
        self.assertEqual(second["provenance"], first["provenance"])
        self.assertIn("2", second["answer"])
        kept = (second["next_context"] or {}).get("last_answer") or {}
        self.assertEqual(kept.get("policy_kind"), "probation_entry")

    def test_new_source_alias_names_canonical_rule(self):
        first = self._ask_with_explain_stub(
            "อุทธรณ์คำสั่งลงโทษต้องยื่นภายในกี่วัน", None
        )
        self.assertEqual(first["status"], "answer")
        second = self._ask_with_explain_stub(
            "มาจากไหน", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        self.assertIn("43", second["answer"])
        self.assertTrue(second["provenance"])

    def test_new_policy_subject_does_not_inherit_appeal_reference(self):
        first = self._ask_with_explain_stub(
            "อุทธรณ์คำสั่งลงโทษต้องยื่นภายในกี่วัน", None
        )
        self.assertEqual(first["status"], "answer")
        self.assertEqual(
            (first["next_context"].get("last_answer") or {}).get("policy_kind"),
            "sanction_appeal_deadline",
        )
        for question in (
            "เกียรตินิยมมาจากกฎข้อไหน",
            "ขยายความเกียรตินิยมหน่อย",
            "เกียรตินิยมอยู่ข้อไหน",
            "ภาคทัณฑ์มาจากไหน",
            "การลาเรียนมาจากข้อไหน",
            "พ้นสภาพมาจากไหน",
        ):
            with self.subTest(question=question):
                with patch.object(
                    main,
                    "answer_previous_followup",
                    wraps=main.answer_previous_followup,
                ) as previous_answer:
                    result = self._ask_with_explain_stub(
                        question, first["next_context"]
                    )
                previous_answer.assert_not_called()
                self.assertIn(
                    result["status"], {"answer", "insufficient_evidence"}
                )
                if result["status"] == "insufficient_evidence":
                    self.assertEqual(result["provenance"], [])

    def test_explicit_course_target_does_not_inherit_previous_answer(self):
        first = self._ask_with_credit_explain_stub(
            "06016414 กี่หน่วย", {"program": "IT", "catalog_key": "it-2565"}
        )
        self.assertEqual(first["status"], "answer")
        for question in ("06016414 มาจากไหน", "ขยายความ 06016414 หน่อย"):
            with self.subTest(question=question):
                with patch.object(
                    main,
                    "answer_previous_followup",
                    wraps=main.answer_previous_followup,
                ) as previous_answer:
                    result = self._ask_with_credit_explain_stub(
                        question, first["next_context"]
                    )
                previous_answer.assert_not_called()
                self.assertIn(
                    result["status"], {"answer", "insufficient_evidence", "error"}
                )

    def test_new_rationale_alias_fails_closed(self):
        first = self._ask_with_explain_stub(
            "อุทธรณ์คำสั่งลงโทษต้องยื่นภายในกี่วัน", None
        )
        for followup in ("เพราะอะไร", "กำหนดแบบนี้เพราะอะไร"):
            with self.subTest(followup=followup):
                second = self._ask_with_explain_stub(
                    followup, first["next_context"]
                )
                self.assertEqual(second["status"], "insufficient_evidence")
                self.assertEqual(second["provenance"], [])

    def test_new_course_explain_alias_regrounds_course(self):
        first = self._ask_with_credit_explain_stub(
            "06016414 กี่หน่วย", {"program": "IT", "catalog_key": "it-2565"}
        )
        self.assertEqual(first["status"], "answer")
        second = self._ask_with_credit_explain_stub(
            "อธิบายอีกที", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        self.assertIn("06016414", second["answer"])
        self.assertNotIn("ภาคทัณฑ์", second["answer"])

    def test_new_ordinal_explain_alias_uses_resolved_course(self):
        search = self._ask_with_credit_explain_stub(
            "มีวิชาเกี่ยวกับ cybersecurity อะไรบ้าง",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        credit = self._ask_with_credit_explain_stub(
            "ตัวแรกกี่หน่วย", search["next_context"]
        )
        self.assertEqual(credit["status"], "answer")
        explained = self._ask_with_credit_explain_stub(
            "อธิบายอีกที", credit["next_context"]
        )
        self.assertEqual(explained["status"], "answer")
        self.assertIn("06016405", explained["answer"])
        self.assertNotIn("06016438", explained["answer"])

    def test_new_detail_alias_preserves_null_placement(self):
        first = self._ask_with_stub_provider(
            "06016438 เรียนตอนไหน",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(first["status"], "answer")
        self.assertIn("06016438", first["answer"])
        second = self._ask_with_explain_stub(
            "ขอรายละเอียดเพิ่มหน่อย", first["next_context"]
        )
        self.assertEqual(second["status"], "answer")
        self.assertIn("06016438", second["answer"])
        self.assertIn(
            "ไม่มีข้อมูลปี/ภาคเรียนที่แน่นอนในหลักสูตร", second["answer"]
        )
        self.assertNotIn("เรียนในปี", second["answer"])

    def test_new_substantive_controls_not_captured(self):
        first = self._probation_answer()
        for question in (
            "ขอรายละเอียดวิชา 06016414",
            "ข้อมูลนี้มาจากวิชาอะไร",
            "หมายถึง 06016414 ใช่ไหม",
            "เพราะอะไรวิชา 06016414 ถึงมี 3 หน่วยกิต",
        ):
            with self.subTest(question=question):
                payload = self._ask_with_explain_stub(
                    question, first["next_context"]
                )
                self.assertNotIn("ภาคทัณฑ์", payload["answer"])

    def test_new_aliases_fresh_fail_closed(self):
        for question in (
            "ขยายอีกหน่อย",
            "อธิบายอีกที",
            "หมายถึงอะไร",
            "ขอรายละเอียดเพิ่มหน่อย",
            "มาจากไหน",
            "ขอดูที่มา",
            "เพราะอะไร",
        ):
            with self.subTest(question=question):
                payload = self._ask_with_explain_stub(question, None)
                self.assertEqual(payload["status"], "insufficient_evidence")
                self.assertEqual(payload["provenance"], [])

    # --- Plan scope + canonical program total (correctness micro-task) ---
    def test_plan_qualified_total_uses_canonical_requirement(self):
        payload = self._ask_with_stub_provider(
            "IT สหกิจรวมกี่หน่วยกิต",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(payload["status"], "answer")
        self.assertIn("129", payload["answer"])
        self.assertTrue(payload["provenance"])

    def test_other_plan_total_is_not_invented(self):
        payload = self._ask_with_stub_provider(
            "IT ไม่สหกิจรวมกี่หน่วยกิต",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(payload["status"], "answer")
        self.assertIn("129", payload["answer"])
        self.assertTrue(payload["provenance"])

    def test_cross_program_plan_totals_use_requirements(self):
        for question, context, expected in (
            ("BIT สหกิจรวมกี่หน่วยกิต",
             {"program": "BIT", "catalog_key": "bit-2565"}, "126"),
            ("DSBA สหกิจรวมกี่หน่วยกิต",
             {"program": "DSBA", "catalog_key": "dsba-2565"}, "132"),
        ):
            with self.subTest(question=question):
                payload = self._ask_with_stub_provider(question, context)
                self.assertEqual(payload["status"], "answer")
                self.assertIn(expected, payload["answer"])
                self.assertTrue(payload["provenance"])

    def test_total_answer_carries_plan_scope_forward(self):
        payload = self._ask_with_stub_provider(
            "IT สหกิจรวมกี่หน่วยกิต",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(payload["status"], "answer")
        followed = self._ask_with_stub_provider(
            "แล้วมีวิชาอะไรบ้าง", payload["next_context"]
        )
        kept = followed["next_context"] or {}
        self.assertEqual(kept.get("program"), "IT")
        self.assertEqual(kept.get("catalog_key"), "it-2565")
        self.assertEqual(kept.get("plan"), "coop")

    def test_fresh_bare_total_fails_closed(self):
        payload = self._ask_with_stub_provider("รวมกี่หน่วย", None)
        self.assertEqual(payload["status"], "insufficient_evidence")
        self.assertEqual(payload["provenance"], [])

    # --- Micro-task 6: bounded conversational semantic topic ---
    def _ml_search(self):
        return self._ask_with_stub_provider(
            "มีวิชาเกี่ยวกับ machine learning อะไรบ้าง",
            {"program": "IT", "catalog_key": "it-2565"},
        )

    def _assert_topic_scoped_or_safe(self, payload, expected_codes):
        # Refinements may answer from canonical evidence or fail closed, but
        # must never drop the topic and answer a broader curriculum question.
        self.assertIn(payload["status"], ("answer", "insufficient_evidence"))
        self.assertNotEqual(payload.get("action"), "invalid_context")
        self.assertEqual(
            (payload["next_context"] or {}).get("semantic_topic"),
            "machine learning",
        )
        if payload["status"] == "answer":
            self.assertTrue(payload["provenance"])
            self.assertTrue(
                any(code in payload["answer"] for code in expected_codes)
            )

    def test_semantic_answer_emits_bounded_topic_state(self):
        payload = self._ml_search()
        self.assertEqual(payload["status"], "answer")
        self.assertTrue(payload["provenance"])
        self.assertEqual(
            (payload["next_context"] or {}).get("semantic_topic"),
            "machine learning",
        )

    def test_topic_survives_safe_refinement_failure(self):
        first = self._ml_search()
        self.assertEqual(first["status"], "answer")

        second = self._ask_with_stub_provider(
            "เอาเฉพาะแผนสหกิจ", first["next_context"]
        )
        self.assertEqual(second["status"], "insufficient_evidence")
        kept = second["next_context"] or {}
        self.assertEqual(kept.get("semantic_topic"), "machine learning")
        self.assertEqual(kept.get("program"), "IT")

    def test_year_refinement_cannot_degrade_into_year_only_dump(self):
        first = self._ml_search()
        second = self._ask_with_stub_provider(
            "มีตัวไหนปี 3 บ้าง", first["next_context"]
        )
        self._assert_topic_scoped_or_safe(second, ("06016460", "06016435"))

    def test_program_refinement_retains_topic(self):
        first = self._ml_search()
        second = self._ask_with_stub_provider("เอาเฉพาะ IT", first["next_context"])
        self.assertEqual((second["next_context"] or {}).get("program"), "IT")
        self._assert_topic_scoped_or_safe(second, ("06016460", "06016435"))

    def test_explicit_new_topic_replaces_old_topic(self):
        first = self._ml_search()
        second = self._ask_with_stub_provider(
            "มีวิชาเกี่ยวกับ cybersecurity อะไรบ้าง", first["next_context"]
        )
        kept = second["next_context"] or {}
        self.assertEqual(kept.get("semantic_topic"), "cybersecurity")
        self.assertNotIn("machine learning", json.dumps(kept))
        if second["status"] == "answer":
            self.assertIn("06016405", second["answer"])

    def test_fresh_session_has_no_semantic_topic(self):
        payload = self._ask_with_stub_provider(
            "ปี 3 มีวิชาอะไรบ้าง", {"program": "IT", "catalog_key": "it-2565"}
        )
        self.assertNotEqual(payload["status"], "error")
        self.assertNotIn("semantic_topic", payload["next_context"] or {})

    def test_malformed_semantic_topic_rejected(self):
        base = {"program": "IT", "catalog_key": "it-2565"}
        for bad_topic in ({"topic": "ml"}, "x" * 121, 123, ""):
            with self.subTest(bad_topic=str(bad_topic)[:20]):
                response = self.client.post(
                    "/api/ask",
                    json={
                        "question": "มีตัวไหนปี 3 บ้าง",
                        "conversation_context": {**base, "semantic_topic": bad_topic},
                    },
                )
                self.assertEqual(response.status_code, 422)

        overlong = run_ask_sql(
            DB_PATH,
            "มีตัวไหนปี 3 บ้าง",
            "IT",
            lambda _prompt: self.fail("no SQL for invalid topic"),
            lambda _prompt: self.fail("no answer for invalid topic"),
            conversation_context={**base, "semantic_topic": "x" * 121},
        )
        self.assertEqual(overlong["status"], "error")

    def test_nonsense_topic_yields_no_fabricated_courses(self):
        payload = self._ask_with_stub_provider(
            "มีวิชาเกี่ยวกับ flurbnax quantum banana อะไรบ้าง",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(payload["status"], "insufficient_evidence")
        self.assertEqual(payload["provenance"], [])
        self.assertNotRegex(payload["answer"], r"0\d{7}")

    # --- Micro-task 5: pre-SQL structural eligibility (single shared rule) ---
    def _ask_with_forbidden_provider(self, question, context):
        """Run POST /api/ask with real ask_sql but a provider that explodes.

        Any model call (SQL generation, answer synthesis, intent) raises out
        of the request, so reaching an assertion proves zero model calls.
        """

        def forbidden(prompt, **options):
            raise AssertionError(
                "no model call allowed for structurally unanswerable queries"
            )

        with (
            patch.object(main, "answer_hard_question", return_value=None),
            patch.object(main, "_lazy_provider", side_effect=forbidden),
            patch.object(
                main, "ask_sql", side_effect=lambda *args, **kwargs: run_ask_sql(*args, **kwargs)
            ),
        ):
            response = self.client.post(
                "/api/ask",
                json={"question": question, "conversation_context": context},
            )
        self.assertEqual(response.status_code, 200, response.json())
        return response.json()

    def test_targetless_placement_rejected_before_sql_generation(self):
        for _ in range(20):
            payload = self._ask_with_forbidden_provider(
                "ตัวนี้เรียนตอนไหน", {"program": "IT", "catalog_key": "it-2565"}
            )
            self.assertEqual(payload["status"], "insufficient_evidence")
            self.assertNotEqual(payload.get("action"), "invalid_sql")
            self.assertEqual(payload["provenance"], [])
            self.assertEqual(
                payload["next_context"],
                {"program": "IT", "catalog_key": "it-2565"},
            )

    def test_targetless_credits_rejected_before_sql_generation(self):
        payload = self._ask_with_forbidden_provider(
            "กี่หน่วยอะ", {"program": "IT", "catalog_key": "it-2565"}
        )
        self.assertEqual(payload["status"], "insufficient_evidence")
        self.assertNotEqual(payload.get("action"), "invalid_sql")
        self.assertEqual(payload["provenance"], [])

    def test_explicit_course_query_still_reaches_normal_path(self):
        def credit_intent(prompt, **options):
            if prompt.startswith(
                "ROLE: You interpret Thai university curriculum questions"
            ):
                return json.dumps(
                    {
                        "intent": "course_credit_query",
                        "proposed_program": None,
                        "proposed_plans": [],
                        "proposed_years": [],
                        "proposed_semesters": [],
                        "course_codes": ["06016414"],
                        "topic": None,
                        "requested_facts": ["course_credit"],
                        "judgement_dimension": None,
                        "unresolved": [],
                    },
                    ensure_ascii=False,
                )
            if "Matching database rows exist" in prompt:
                return "พบข้อมูลรายวิชาที่ตรงกับคำถาม"
            return (
                "SELECT course_code, program, plan_key, year, semester "
                "FROM v_plan_courses"
            )

        with (
            patch.object(main, "answer_hard_question", return_value=None),
            patch.object(main, "_lazy_provider", side_effect=credit_intent),
            patch.object(
                main, "ask_sql", side_effect=lambda *args, **kwargs: run_ask_sql(*args, **kwargs)
            ),
        ):
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "06016414 กี่หน่วย",
                    "conversation_context": {
                        "program": "IT",
                        "catalog_key": "it-2565",
                    },
                },
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "answer")
        self.assertIn("06016414", payload["answer"])
        self.assertIn("3", payload["answer"])
        self.assertTrue(payload["provenance"])

    def test_context_resolved_followup_still_reaches_normal_path(self):
        first = self._ask_with_stub_provider(
            "06016414 คือวิชาอะไร",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(first["status"], "answer")

        second = self._ask_with_stub_provider("มันกี่หน่วย", first["next_context"])
        self.assertEqual(second["status"], "answer")
        self.assertIn("06016414", second["answer"])

    def test_semester_aggregate_not_rejected_as_targetless(self):
        payload = self._ask_with_stub_provider(
            "IT ปี1เทอม1 รวมกี่หน่วย",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(payload["status"], "answer")
        self.assertTrue(payload["provenance"])
        self.assertEqual(payload["next_context"]["years"], [1])
        self.assertEqual(payload["next_context"]["semesters"], [1])

    def test_program_total_not_rejected_as_targetless(self):
        payload = self._ask_with_stub_provider(
            "IT แผนสหกิจรวมทั้งหมดกี่หน่วยกิต",
            {"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(payload["status"], "answer")
        self.assertIn("129", payload["answer"])

    def test_genuine_invalid_sql_for_valid_query_remains_an_error(self):
        def garbage_sql(prompt, **options):
            return "THIS IS NOT SQL"

        with (
            patch.object(main, "answer_hard_question", return_value=None),
            patch.object(main, "_lazy_provider", side_effect=garbage_sql),
            patch.object(
                main, "ask_sql", side_effect=lambda *args, **kwargs: run_ask_sql(*args, **kwargs)
            ),
        ):
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "06016414 กี่หน่วย",
                    "conversation_context": {
                        "program": "IT",
                        "catalog_key": "it-2565",
                    },
                },
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["action"], "invalid_sql")


if __name__ == "__main__":
    unittest.main()
