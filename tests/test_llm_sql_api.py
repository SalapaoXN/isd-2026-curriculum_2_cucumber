import unittest
import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main
from backend.hard_qa import HARD_INTERPRETATION_RESPONSE_JSON_SCHEMA, answer_hard_question
from backend.llm_sql_qa import ask_sql as run_ask_sql
from rag import qa as rag_qa
from rag.query_spec import detect_surface_operations, parse_query_spec


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
                    self.assertEqual(
                        payload["next_context"],
                        {
                            "program": "DSBA",
                            "catalog_key": catalog_key,
                            "years": [1],
                            "semesters": [1],
                            "operations": ["list"],
                        },
                    )

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

    def test_hard_seven_term_without_plan_returns_each_canonical_plan(self):
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
        self.assertEqual(
            payload["next_context"],
            {"catalog_key": "dsba-2565", "program": "DSBA"},
        )
        self.assertEqual(
            [item["plan"] for item in payload["plan_results"]],
            ["coop", "no_coop"],
        )
        self.assertEqual(
            [item["status"] for item in payload["plan_results"]],
            ["incomplete_evidence", "incomplete_evidence"],
        )
        for item in payload["plan_results"]:
            with self.subTest(plan=item["plan"]):
                self.assertTrue(item["provenance"])
                self.assertIn("ปี 1 เทอม 1:", item["answer"])
                self.assertIn("ช่องวิชาเลือกที่ยังไม่ระบุวิชาจริง", item["answer"])
                self.assertIn("โครงร่าง 7 เทอมนี้ไม่ใช่ข้อพิสูจน์ว่าครบเงื่อนไขจบ", item["answer"])
                self.assertNotIn("H2 could not establish a complete mandatory course set", item["answer"])
                for placeholder in ("06026xxx", "90644xxx", "9064xxxx", "xxxxxxxx"):
                    self.assertNotIn(placeholder, item["answer"])
        self.assertEqual(payload["status"], "incomplete_evidence")
        self.assertIn("แผน coop", payload["answer"])
        self.assertIn("แผน no_coop", payload["answer"])
        self.assertNotIn("เลือกแผนหลักสูตรที่ต้องการตรวจสอบ", payload["answer"])
        self.assertEqual(self.sql_service.call_count, 0)
        provider.assert_called_once()

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
