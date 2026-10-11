import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest import mock
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main, rag_service
from backend.schemas import AnswerDraft, ChatTurn, CurriculumScope, GeneralDraft, RouteDecision


class RAGServiceTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close()

    def test_general_conversation_skips_retrieval(self):
        def generate(_prompt, _payload, schema):
            return GeneralDraft(answer="Hello! How can I help with your curriculum?")

        with patch.object(rag_service, "_classify_use_rag", return_value=False), \
             patch.object(rag_service, "_generate", side_effect=generate), \
             patch.object(rag_service, "_search", side_effect=AssertionError("retrieval should be skipped")):
            result = rag_service.answer("Hello", [])
        self.assertEqual(result["status"], "general")
        self.assertEqual(result["sources"], [])
        self.assertFalse(result["database_searched"])
        self.assertGreaterEqual(result["response_time_ms"], 0)

    def test_success_logs_route_and_result_without_question_text(self):
        secret_question = "private question text 0600"
        with patch.object(rag_service, "_classify_use_rag", return_value=False), \
             patch.object(rag_service, "_generate", return_value=GeneralDraft(answer="Hello")), \
             self.assertLogs(rag_service._logger, level="INFO") as captured:
            result = rag_service.answer(secret_question, [])
        log_text = "\n".join(captured.output)
        self.assertNotIn(secret_question, log_text)
        self.assertIn("event=classified use_rag=False", log_text)
        self.assertIn("database_searched=False", log_text)
        self.assertIn(result["request_id"], log_text)

    def test_ambiguous_request_returns_clarification_without_search(self):
        decision = RouteDecision(intent="clarification", use_rag=False,
                                 standalone_query="Which course?", needs_clarification=True,
                                 clarification_question="Which program and curriculum year do you mean?")
        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", return_value=decision), \
             patch.object(rag_service, "_search", side_effect=AssertionError("retrieval should be skipped")):
            result = rag_service.answer("What does that require?", [])
        self.assertEqual(result["status"], "clarification_required")
        self.assertIn("Which program", result["answer"])
        self.assertFalse(result["database_searched"])

    def test_missing_edition_clarification_includes_selectable_program_editions(self):
        decision = RouteDecision(
            intent="course_question", use_rag=False, standalone_query="DSBA year 1 courses",
            needs_clarification=True, clarification_question="Which curriculum edition do you mean?",
            program_code=None,
        )
        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", return_value=decision), \
             patch("backend.curriculum.filters", side_effect=[
                 {"programs": ["DSBA", "AIT"]},
                 {"academic_years": ["2565", "2560"]},
             ]):
            result = rag_service.answer("DSBA year 1 semester 1 courses", [])

        self.assertEqual(result["status"], "clarification_required")
        self.assertIn("DSBA", result["answer"])
        self.assertEqual(result["options"], ["2565", "2560"])
        self.assertGreaterEqual(result["response_time_ms"], 0)

    def test_course_answer_citations_are_limited_to_retrieved_chunks(self):
        decision = RouteDecision(intent="course_question", use_rag=True,
                                 standalone_query="IT year one courses", needs_clarification=False,
                                 clarification_question=None)
        chunks = [
            {"chunk_id": "real-chunk", "text": "Year 1 includes course A", "program": "IT", "distance": 0.4,
             "plan": "coop", "course_code": "ABC100", "provenance": [{"source_page": 9,
             "source_filename": "it-plan.pdf"}]},
            {"chunk_id": "unselected-chunk", "text": "Other course information", "program": "IT", "distance": 0.5},
        ]

        def generate(_prompt, _payload, schema):
            if schema is RouteDecision:
                return decision
            return AnswerDraft(sufficient=True, answer="Year 1 includes course A.", source_ids=["real-chunk"])

        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", side_effect=generate), \
             patch.object(rag_service, "_search", return_value=chunks):
            result = rag_service.answer("What courses are in IT year one?", [])
        self.assertEqual(result["status"], "answer")
        self.assertTrue(result["database_searched"])
        self.assertEqual([source["chunk_id"] for source in result["sources"]], ["real-chunk"])
        self.assertEqual(result["sources"][0]["source_page"], 9)

    def test_exhaustive_course_list_uses_structured_database_and_formats_every_item(self):
        decision = RouteDecision(
            intent="course_question", use_rag=True, standalone_query="IT 2565 year 1 semester 1",
            needs_clarification=False, clarification_question=None, exhaustive_course_list=True,
            program_code="IT", academic_year="2565", study_year=1, semester=1,
        )
        courses = [{
            "course_id": index, "course_code": code, "name_th": None, "name_en": name,
            "credits": "3(3-0-6)", "academic_year": "2565", "program_code": "IT",
            "plans": ["coop", "no_coop"], "sources": [f"it2565_page_{index}.png · p.{index}"],
        } for index, code, name in ((1, "IT101", "Course One"), (2, "IT102", "Course Two"))]

        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", return_value=decision), \
             patch("backend.curriculum.list_courses", return_value={"items": courses, "total": 2}), \
             patch.object(rag_service, "_search", side_effect=AssertionError("course lists use structured retrieval")):
            result = rag_service.answer("List all IT 2565 year 1 semester 1 courses", [])

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["retrieval_count"], 2)
        self.assertIn("2 subjects", result["answer"])
        self.assertIn("1. IT101 — Course One — 3(3-0-6) credits", result["answer"])
        self.assertIn("2. IT102 — Course Two — 3(3-0-6) credits", result["answer"])
        self.assertEqual(len(result["sources"]), 2)

    def test_course_detail_expands_multiple_flexible_year_semester_terms(self):
        decision = RouteDecision(
            intent="course_question", use_rag=True,
            standalone_query="DATA PIPELINE ARCHITECTURE year and semester",
            needs_clarification=False, clarification_question=None, course_detail=True,
            course_name="DATA PIPELINE ARCHITECTURE", program_code="DSBA", academic_year="2565",
        )
        course = {
            "course_id": 39, "course_code": "06026239", "name_th": "สถาปัตยกรรมข้อมูลแบบไปป์ไลน์",
            "name_en": "DATA PIPELINE ARCHITECTURE", "program_code": "DSBA", "academic_year": "2565",
            "credits": "3(3-0-6)",
            "plans": ["coop", "no_coop"], "placements": ["3/1", " 3/2", " 4/1"],
            "sources": ["dsba2565.pdf · p.12"],
        }
        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", return_value=decision), \
             patch("backend.curriculum.list_courses", return_value={"items": [course], "total": 1}) as query, \
             patch.object(rag_service, "_search", side_effect=AssertionError("course details use structured lookup")):
            result = rag_service.answer("DATA PIPELINE ARCHITECTURE ต้องเรียนปีไหน เทอมไหน", [])

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["retrieval_count"], 1)
        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("ชั้นปีที่ 3 · ภาคเรียนที่ 1", result["answer"])
        self.assertIn("ชั้นปีที่ 3 · ภาคเรียนที่ 2", result["answer"])
        self.assertIn("ชั้นปีที่ 4 · ภาคเรียนที่ 1", result["answer"])
        self.assertIn("3(3-0-6)", result["answer"])
        self.assertEqual(query.call_args.kwargs["course_code"], "")

    def test_course_detail_uses_exact_code_and_returns_credits_and_placement(self):
        decision = RouteDecision(
            intent="course_question", use_rag=True, standalone_query="credits for 06046401",
            needs_clarification=False, clarification_question=None, course_detail=True,
            course_code="06046401", program_code="AIT", academic_year="2566",
        )
        course = {"course_id": 1, "course_code": "06046401", "name_en": "CALCULUS 2",
                  "name_th": "แคลคูลัส 2", "credits": "3(3-0-6)", "program_code": "AIT",
                  "academic_year": "2566", "plans": ["default"],
                  "placements": ["Year 1 · Semester 2"], "sources": ["ait.pdf · p.10"]}
        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", return_value=decision), \
             patch("backend.curriculum.list_courses", return_value={"items": [course], "total": 1}) as query:
            result = rag_service.answer("วิชา 06046401 กี่หน่วยกิต", [])
        self.assertIn("3(3-0-6)", result["answer"])
        self.assertIn("CALCULUS 2", result["answer"])
        self.assertIn("ชั้นปีที่ 1 · ภาคเรียนที่ 2", result["answer"])
        self.assertEqual(query.call_args.kwargs["course_code"], "06046401")

    def test_inverse_prerequisite_route_lists_courses_requiring_named_course(self):
        decision = RouteDecision(
            intent="course_question", use_rag=True,
            standalone_query="Courses requiring Calculus 1", needs_clarification=False,
            clarification_question=None, prerequisite_question=True,
            course_code="06046400",
            program_code="AIT", academic_year="2566",
        )
        dependent = {"course_id": 2, "course_code": "06046401", "name_en": "CALCULUS 2",
                     "name_th": "แคลคูลัส 2", "program_code": "AIT", "academic_year": "2566",
                     "plans": ["default"], "sources": ["ait.pdf · p.11"]}
        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", return_value=decision), \
             patch("backend.curriculum.courses_requiring", return_value={"items": [dependent], "total": 1}):
            result = rag_service.answer("แคลคูลัส 1 ต้องเรียนก่อนวิชาอะไร", [])
        self.assertIn("06046401", result["answer"])
        self.assertIn("CALCULUS 2", result["answer"])
        self.assertEqual(len(result["sources"]), 1)

    def test_inverse_prerequisite_returns_common_answer_when_unscoped_matches_agree(self):
        decision = RouteDecision(
            intent="course_question", use_rag=True,
            standalone_query="Courses requiring CALCULUS 1", needs_clarification=False,
            clarification_question=None, prerequisite_question=True,
            prerequisite_direction="courses_requiring", course_name="CALCULUS 1",
        )
        candidates = [
            {"course_code": "06046400", "name_en": "CALCULUS 1", "name_th": "แคลคูลัส 1",
             "program_code": "AIT", "academic_year": "2566"},
            {"course_code": "06026200", "name_en": "CALCULUS 1", "name_th": "แคลคูลัส 1",
             "program_code": "DSBA", "academic_year": "2565"},
        ]
        dependents = [
            {"course_id": 7, "course_code": "06046401", "name_en": "CALCULUS 2",
             "name_th": "แคลคูลัส 2", "program_code": "AIT", "academic_year": "2566",
             "plans": ["default"], "sources": ["ait.pdf · p.11"]},
            {"course_id": 8, "course_code": "06026201", "name_en": "CALCULUS 2",
             "name_th": "แคลคูลัส 2", "program_code": "DSBA", "academic_year": "2565",
             "plans": ["coop", "no_coop"], "sources": ["dsba.pdf · p.10"]},
        ]
        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", return_value=decision), \
             patch("backend.curriculum.list_courses", return_value={"items": candidates, "total": 2}), \
             patch("backend.curriculum.courses_requiring", side_effect=[
                 {"items": [dependents[0]], "total": 1},
                 {"items": [dependents[1]], "total": 1},
             ]):
            result = rag_service.answer("วิชาแคลคูลัส 1 ต้องเรียนก่อนวิชาอะไร", [])
        self.assertEqual(result["status"], "answer")
        self.assertIn("CALCULUS 2", result["answer"])
        self.assertIn("06046401", result["answer"])
        self.assertIn("06026201", result["answer"])
        self.assertEqual(len(result["sources"]), 2)

    def test_program_requirement_route_returns_scoped_total_and_source(self):
        decision = RouteDecision(
            intent="course_question", use_rag=True,
            standalone_query="IT curriculum 2565 total program credits",
            needs_clarification=False, clarification_question=None,
            program_requirement=True, program_code="IT", academic_year="2565",
        )
        requirement = {"requirement_id": 8, "program_code": "IT", "academic_year": "2565",
                       "requirement_type": "total_program_credits", "operator": "=", "value": 129,
                       "unit": "credits", "sources": [{"source_filename": "it-overview.pdf",
                       "source_page": 1, "source_uri": None, "source_locator": None}]}
        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", return_value=decision), \
             patch("backend.curriculum.program_requirements", return_value={"items": [requirement], "total": 1}):
            result = rag_service.answer("IT 2565 ต้องเรียนกี่หน่วยกิตจึงจบ", [])
        self.assertIn("129 หน่วยกิต", result["answer"])
        self.assertEqual(result["sources"][0]["source_page"], 1)

    def test_program_requirement_comparison_uses_each_scoped_total(self):
        decision = RouteDecision(
            intent="course_question", use_rag=True,
            standalone_query="Compare IT 2565 and DSBA 2565 graduation credits",
            needs_clarification=False, clarification_question=None,
            compare_program_requirements=True,
            left_scope=CurriculumScope(program_code="IT", academic_year="2565"),
            right_scope=CurriculumScope(program_code="DSBA", academic_year="2565"),
        )
        it = {"requirement_id": 1, "program_code": "IT", "academic_year": "2565",
              "value": 129, "unit": "credits", "sources": [{"source_filename": "it.pdf", "source_page": 1}]}
        dsba = {"requirement_id": 2, "program_code": "DSBA", "academic_year": "2565",
                "value": 132, "unit": "credits", "sources": [{"source_filename": "dsba.pdf", "source_page": 2}]}
        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", return_value=decision), \
             patch("backend.curriculum.program_requirements", side_effect=[
                 {"items": [it], "total": 1}, {"items": [dsba], "total": 1},
             ]):
            result = rag_service.answer("เปรียบเทียบหน่วยกิตจบ IT กับ DSBA ปี 2565", [])
        self.assertIn("IT 2565: 129 หน่วยกิต", result["answer"])
        self.assertIn("DSBA 2565: 132 หน่วยกิต", result["answer"])
        self.assertTrue(result["answer"].splitlines()[-1].endswith("3 หน่วยกิต"))
        self.assertEqual(len(result["sources"]), 2)

    def test_course_detail_followup_preserves_history_for_resolver(self):
        decision = RouteDecision(
            intent="course_question", use_rag=True,
            standalone_query="AIT 2566 Calculus 2 credits",
            needs_clarification=False, clarification_question=None, course_detail=True,
            course_code="06046401", program_code="AIT", academic_year="2566",
        )
        course = {"course_id": 1, "course_code": "06046401", "name_en": "CALCULUS 2",
                  "credits": "3(3-0-6)", "program_code": "AIT", "academic_year": "2566",
                  "plans": ["default"], "placements": ["Year 1 · Semester 2"], "sources": []}
        def resolve(_prompt, payload, schema):
            self.assertEqual(schema, RouteDecision)
            self.assertEqual(payload["history"][-1]["content"], "วิชาแคลคูลัส 2 ของ AIT คืออะไร")
            return decision
        history = [ChatTurn(role="user", content="วิชาแคลคูลัส 2 ของ AIT คืออะไร")]
        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", side_effect=resolve), \
             patch("backend.curriculum.list_courses", return_value={"items": [course], "total": 1}):
            result = rag_service.answer("แล้วหน่วยกิตล่ะ", history)
        self.assertIn("3(3-0-6)", result["answer"])

    def test_prerequisite_question_resolves_prerequisite_code_to_course_name(self):
        decision = RouteDecision(
            intent="course_question", use_rag=True,
            standalone_query="Prerequisite for INFRASTRUCTURE SYSTEMS AND SERVICES",
            needs_clarification=False, clarification_question=None,
            prerequisite_question=False, course_code="06016420", program_code="IT", plan="ไม่สหกิจ",
        )
        target = {
            "course_id": 420, "course_code": "06016420", "name_en": "INFRASTRUCTURE SYSTEMS AND SERVICES",
            "name_th": "ระบบโครงสร้างพื้นฐานและการบริการ", "prerequisite_text": "06016413",
            "program_code": "IT", "academic_year": "2565", "plans": ["coop", "no_coop"],
            "sources": ["it2565_page_035.png · p.35"],
        }
        prerequisite = {
            "course_id": 413, "course_code": "06016413", "name_en": "INTRODUCTION TO NETWORK SYSTEMS",
            "name_th": "การแนะนำระบบเครือข่าย", "prerequisite_text": "ไม่มี",
            "program_code": "IT", "academic_year": "2565", "plans": ["coop", "no_coop"],
            "sources": ["it2565_page_034.png · p.34"],
        }
        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", return_value=decision), \
             patch("backend.curriculum.list_courses", side_effect=[
                 {"items": [target], "total": 1}, {"items": [prerequisite], "total": 1},
             ]) as query, \
             patch.object(rag_service, "_search", side_effect=AssertionError("prerequisites use structured lookup")):
            result = rag_service.answer(
                "ก่อนลง INFRASTRUCTURE SYSTEMS AND SERVICES (06016420) ใน IT แบบไม่สหกิจ ต้องผ่านวิชาอะไรมาก่อน?", []
            )

        self.assertEqual(query.call_count, 2)
        self.assertEqual(query.call_args_list[0].kwargs["plan"], "no_coop")
        self.assertEqual(result["status"], "answer")
        self.assertIn("06016413 — INTRODUCTION TO NETWORK SYSTEMS", result["answer"])
        self.assertIn("no_coop", result["answer"])
        self.assertEqual(len(result["sources"]), 2)

    def test_flexible_placement_parser_handles_comma_separated_terms(self):
        self.assertEqual(
            rag_service._expand_placement_terms(["3/1, 3/2, 4/1"]),
            ["Year 3 · Semester 1", "Year 3 · Semester 2", "Year 4 · Semester 1"],
        )

    def test_institutional_rule_question_uses_rules_corpus_and_preserves_citation(self):
        decision = RouteDecision(
            intent="course_question", use_rag=True, standalone_query="GPA 1.9 probation",
            needs_clarification=False, clarification_question=None,
            institution_rule=False, rule_query=None,
        )
        rule_chunk = {
            "chunk_id": "rule:22", "chunk_type": "rule", "rule_id": "rule:22",
            "section_number": "22", "category": "Academic probation",
            "source_verified": True,
            "text": "Cumulative GPA below 2.00 results in probation.", "distance": 0.1,
            "provenance": [{"source_filename": "academic-rules.pdf", "source_page": 7}],
        }

        def generate(_prompt, _payload, schema):
            if schema is RouteDecision:
                return decision
            return AnswerDraft(sufficient=True, answer="A cumulative GPA below 2.00 results in probation.",
                               source_ids=["rule:22"])

        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", side_effect=generate), \
             patch.object(rag_service, "_search_rules", return_value=[rule_chunk]), \
             patch.object(rag_service, "_search", side_effect=AssertionError("rules use the rules corpus")):
            result = rag_service.answer("GPA 1.9 will I be on probation?", [])

        self.assertEqual(result["status"], "answer")
        self.assertTrue(result["database_searched"])
        self.assertEqual(result["retrieval_count"], 1)
        self.assertEqual(result["sources"][0]["rule_id"], "rule:22")
        self.assertEqual(result["sources"][0]["source_page"], 7)

    def test_rule_embedding_corpus_is_loaded_from_sqlite_rule_and_policy_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "curriculum.db"
            with closing(sqlite3.connect(database)) as db:
                db.executescript("""
                    CREATE TABLE regulation_rules(
                        rule_id TEXT, section_number TEXT, category TEXT, rule_text TEXT,
                        references_json TEXT);
                    CREATE TABLE policy_facts(
                        fact_id INTEGER, source_rule_id TEXT, category TEXT, fact_key TEXT,
                        operator TEXT, value NUMERIC, unit TEXT, condition TEXT, context TEXT,
                        verification_status TEXT);
                """)
                db.execute("INSERT INTO regulation_rules VALUES (?,?,?,?,?)", (
                    "rule:22", "22", "Probation",
                    "Cumulative GPA below 2.00 results in probation.",
                    json.dumps({"source_provenance": [{"source_filename": "rules.pdf",
                                                       "source_page": 7, "source_verified": True}]}),
                ))
                db.execute("INSERT INTO policy_facts VALUES (?,?,?,?,?,?,?,?,?,?)", (
                    1, "rule:22", "Probation", "cumulative GPA", "below", 2.0,
                    None, "threshold", None, "verified",
                ))
                db.commit()

            rag_service._rule_index.cache_clear()
            vectors = rag_service.np.asarray([[1.0, 0.0]], dtype=rag_service.np.float32)
            with patch.object(rag_service, "INDEX_PATH", database), \
                 patch.object(rag_service, "embed_texts", return_value=vectors):
                chunks, matrix = rag_service._rule_index()
            rag_service._rule_index.cache_clear()

        self.assertEqual(chunks[0]["rule_id"], "rule:22")
        self.assertIn("Cumulative GPA below 2.00", chunks[0]["text"])
        self.assertIn("cumulative GPA threshold below 2", chunks[0]["text"])
        self.assertTrue(chunks[0]["source_verified"])
        self.assertEqual(chunks[0]["provenance"][0]["source_page"], 7)
        self.assertEqual(matrix.shape, (1, 2))

    def test_curriculum_comparison_matches_codes_and_normalized_names_with_both_sources(self):
        decision = RouteDecision(
            intent="course_question", use_rag=True, standalone_query="shared curriculum courses",
            needs_clarification=False, clarification_question=None, compare_curricula=True,
            left_scope=CurriculumScope(program_code="DSBA", academic_year="2565", study_year=1, semester=1),
            right_scope=CurriculumScope(program_code="AIT", academic_year="2566", study_year=1, semester=1),
            comparison_operation="intersection",
        )
        left = [
            {"course_id": 1, "course_code": "SAME01", "name_en": "Shared Course", "name_th": None,
             "credits": "3", "program_code": "DSBA", "academic_year": "2565", "plans": ["coop"],
             "sources": ["dsba.pdf · p.10"]},
            {"course_id": 2, "course_code": "OLD01", "name_en": "Linear Algebra", "name_th": None,
             "credits": "3", "program_code": "DSBA", "academic_year": "2565", "plans": ["coop"],
             "sources": ["dsba.pdf · p.11"]},
        ]
        right = [
            {"course_id": 3, "course_code": "SAME01", "name_en": "Shared Course", "name_th": None,
             "credits": "3", "program_code": "AIT", "academic_year": "2566", "plans": ["default"],
             "sources": ["ait.pdf · p.20"]},
            {"course_id": 4, "course_code": "NEW01", "name_en": "LINEAR   ALGEBRA ALGEBRA", "name_th": None,
             "credits": "3", "program_code": "AIT", "academic_year": "2566", "plans": ["default"],
             "sources": ["ait.pdf · p.21"]},
        ]

        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", return_value=decision), \
             patch("backend.curriculum.list_courses", side_effect=[
                 {"items": left, "total": 2}, {"items": right, "total": 2},
             ]) as query, \
             patch.object(rag_service, "_search", side_effect=AssertionError("comparisons use structured retrieval")):
            result = rag_service.answer("Which courses are shared by these curricula?", [])

        self.assertEqual(query.call_count, 2)
        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["retrieval_count"], 4)
        self.assertEqual(len(result["sources"]), 4)
        self.assertIn("Shared Course (same course code)", result["answer"])
        self.assertIn("Linear Algebra (same normalized name)", result["answer"])
        self.assertNotIn("LINEAR ALGEBRA ALGEBRA", result["answer"])
        self.assertIn("DSBA 2565 Year 1 Semester 1", result["answer"])
        self.assertIn("AIT 2566 Year 1 Semester 1", result["answer"])

    def test_top_k_includes_all_chunks_tied_at_the_boundary(self):
        distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6, 0.7]
        calls = []

        def fetch(count):
            calls.append(count)
            return [{"chunk_id": str(index), "distance": distance}
                    for index, distance in enumerate(distances[:count])]

        hits = rag_service._top_k_with_ties(fetch, limit=6)
        self.assertEqual(len(hits), 8)
        self.assertEqual(calls, [7, 14])

    def test_unsupported_answer_is_returned_without_citations(self):
        decision = RouteDecision(intent="course_question", use_rag=True,
                                 standalone_query="Scholarships for IT", needs_clarification=False,
                                 clarification_question=None)
        def generate(_prompt, _payload, schema):
            if schema is RouteDecision:
                return decision
            return AnswerDraft(sufficient=False, answer="The indexed sources do not provide scholarship details.", source_ids=[])

        with patch.object(rag_service, "_classify_use_rag", return_value=True), \
             patch.object(rag_service, "_generate", side_effect=generate), \
             patch.object(rag_service, "_search", return_value=[{"chunk_id": "curriculum", "text": "course list", "distance": 0.2}]):
            result = rag_service.answer("What scholarships are available?", [])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["sources"], [])
        self.assertTrue(result["database_searched"])

    def test_invalid_classifier_output_fails_safe(self):
        with patch.object(rag_service, "_classify_use_rag", side_effect=ValueError("invalid output")), \
             patch.object(rag_service, "_search", side_effect=AssertionError("must not retrieve")):
            with self.assertRaises(Exception) as error:
                rag_service.answer("What is the credit requirement?", [])
        self.assertEqual(error.exception.status_code, 503)

    def test_mini_classifier_returns_only_a_valid_boolean(self):
        provider = mock.Mock(return_value="true\n")
        with patch.object(rag_service, "_classifier_provider", return_value=provider):
            result = rag_service._classify_use_rag("How many credits?", [])
        self.assertIs(result, True)
        args, kwargs = provider.call_args
        self.assertEqual(kwargs["response_mime_type"], "text/plain")
        self.assertIn('"latest_message": "How many credits?"', args[0])

    def test_classifier_uses_a_distinct_mini_model(self):
        rag_service._classifier_provider.cache_clear()
        fake_provider = mock.Mock()
        with patch.object(rag_service, "make_gemini_callable", return_value=fake_provider) as factory:
            self.assertIs(rag_service._classifier_provider(), fake_provider)
        factory.assert_called_once_with("gemini-3.1-flash-lite")
        rag_service._classifier_provider.cache_clear()

    def test_mini_classifier_rejects_non_boolean_output(self):
        for value in ("probably true", '{"use_rag": true}', "", "1"):
            with self.subTest(value=value):
                provider = mock.Mock(return_value=value)
                with patch.object(rag_service, "_classifier_provider", return_value=provider):
                    with self.assertRaises(ValueError):
                        rag_service._classify_use_rag("Hello", [])

    def test_missing_index_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(rag_service, "INDEX_PATH", Path(directory) / "missing.db"):
            with self.assertRaises(Exception) as error:
                rag_service._search("course query")
        self.assertEqual(error.exception.status_code, 503)
        self.assertIn("rag.build_index", str(error.exception.detail))

    def test_health_and_chat_route_contract(self):
        with patch("backend.main.health", return_value={"status": "ok", "database_ready": True}), \
             patch("backend.main.answer", return_value={"request_id": "req123", "status": "general", "answer": "Hello", "sources": []}):
            health = self.client.get("/api/health")
            answer = self.client.post("/api/chat", json={"question": "Hello", "history": []})
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(answer.json()["answer"], "Hello")
        self.assertEqual(answer.status_code, 200)
        self.assertEqual(answer.headers["X-Request-ID"], "req123")
        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        self.assertIn("Cucumber", root.text)

    def test_chat_request_rejects_whitespace_only_question(self):
        response = self.client.post("/api/chat", json={"question": "   ", "history": []})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
