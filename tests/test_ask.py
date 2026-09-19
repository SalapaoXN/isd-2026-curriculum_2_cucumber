import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import ask
from rag.grounded_answer import GroundedAnswerResult, GroundedClaim


def _response(route="structured"):
    if route == "structured":
        result = {
            "columns": ["course_code", "source_page"],
            "rows": [("06000001", 12)],
            "provenance": [
                {
                    "program": "IT",
                    "source_filename": "it_page_012.png",
                    "source_page": 12,
                }
            ],
        }
    elif route == "semantic":
        result = [
            {
                "chunk_id": "chunk-1",
                "text": "course evidence",
                "source_filename": "it_page_012.png",
                "source_page": [12],
            }
        ]
    else:
        result = {
            "sql_rows": [
                {
                    "provenance": [
                        {
                            "program": "IT",
                            "source_filename": "it_page_012.png",
                            "source_page": 12,
                        }
                    ]
                }
            ],
            "retrieved_chunks": [],
        }
    return {"route": route, "result": result, "final_answer": "คำตอบ"}


class AskCliTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "curriculum.db"
        self.db_path.write_bytes(b"fixture")
        self.provider = lambda _prompt: "คำตอบ"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _patch_runtime(self, response=None):
        return patch.multiple(
            ask,
            DEFAULT_CURRICULUM_DB_PATH=self.db_path,
            make_gemini_callable=lambda: self.provider,
            answer_question_once=lambda *args, **kwargs: response or _response(),
        )

    def test_one_shot_uses_runtime_db_and_prints_answer_and_sources(self):
        response = _response()
        with self._patch_runtime(response), patch.object(
            ask, "answer_question_once", return_value=response
        ) as run_once:
            output = io.StringIO()
            with redirect_stdout(output):
                status = ask.main(["IT course question"])

        self.assertEqual(status, 0)
        run_once.assert_called_once_with(
            self.db_path,
            "IT course question",
            structured_model_callable=self.provider,
            top_k=10,
            answer_model_callable=self.provider,
            intent_model_callable=self.provider,
        )
        text = output.getvalue()
        self.assertIn("ถาม: IT course question", text)
        self.assertIn("ตอบ: คำตอบ", text)
        self.assertIn("แหล่งข้อมูล:\nเล่มหลักสูตร: IT\nหน้า: 12", text)
        self.assertNotIn(", หน้า 12", text)
        self.assertNotIn(".png", text)

    def test_typed_result_provenance_is_displayed_from_grounded_answer(self):
        provenance = (
            {
                "program": "IT",
                "source_filename": "it_page_012.png",
                "source_page": 12,
            },
        )
        claim = GroundedClaim("claim_001", "count", value=1, provenance=provenance)
        result = GroundedAnswerResult(
            "answer",
            "deterministic",
            "คำตอบ",
            (claim,),
            provenance,
        )

        self.assertEqual(
            ask._format_sources({"route": None, "result": result}),
            "เล่มหลักสูตร: IT\nหน้า: 12",
        )
    def test_typed_final_answer_containing_provenance_is_not_removed(self):
        provenance = (
            {
                "program": "IT",
                "source_filename": "it_page_012.png",
                "source_page": 12,
            },
        )
        claim = GroundedClaim("claim_001", "describe", value="grounded", provenance=provenance)
        result = GroundedAnswerResult(
            "answer",
            "deterministic",
            "คำตอบที่มี provenance: สำคัญ",
            (claim,),
            provenance,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            ask._print_result(
                "question",
                {"route": None, "result": result, "final_answer": result.final_answer},
                show_question=False,
            )

        self.assertIn("ตอบ: คำตอบที่มี provenance: สำคัญ", output.getvalue())
        self.assertIn(
            "แหล่งข้อมูล:\nเล่มหลักสูตร: IT\nหน้า: 12", output.getvalue()
        )

    def test_interactive_mode_handles_multiple_questions_and_quit(self):
        responses = [_response("semantic"), _response("hybrid")]
        with self._patch_runtime(), patch.object(
            ask, "answer_question_once", side_effect=responses
        ) as run_once, patch(
            "builtins.input", side_effect=["คำถามแรก", "คำถามสอง", "quit"]
        ), redirect_stdout(io.StringIO()) as output:
            status = ask.main([])

        self.assertEqual(status, 0)
        self.assertEqual(run_once.call_count, 2)
        for call in run_once.call_args_list:
            self.assertEqual(
                call.kwargs.get("intent_model_callable"), self.provider
            )
            self.assertEqual(
                call.kwargs.get("structured_model_callable"), self.provider
            )
        self.assertEqual(output.getvalue().count("ตอบ: คำตอบ"), 2)

    def test_interactive_exit_and_eof_are_clean(self):
        with self._patch_runtime(), patch("builtins.input", return_value="exit"):
            self.assertEqual(ask.main([]), 0)
        with self._patch_runtime(), patch("builtins.input", side_effect=EOFError):
            self.assertEqual(ask.main([]), 0)

    def test_missing_runtime_db_gives_build_instruction_without_provider(self):
        missing = Path(self.temp_dir.name) / "missing.db"
        error = io.StringIO()
        with patch.object(ask, "DEFAULT_CURRICULUM_DB_PATH", missing), patch.object(
            ask, "make_gemini_callable"
        ) as provider:
            with redirect_stderr(error):
                status = ask.main(["question"])

        self.assertEqual(status, 1)
        self.assertIn("python -m rag.build_index", error.getvalue())
        provider.assert_not_called()

    def test_routes_are_preserved_and_user_never_selects_one(self):
        for route in ("structured", "semantic", "hybrid"):
            response = _response(route)
            with self._patch_runtime(), patch.object(
                ask, "answer_question_once", return_value=response
            ) as run_once:
                self.assertEqual(ask.main(["question"]), 0)
                self.assertEqual(run_once.return_value["route"], route)

    def test_fallback_is_displayed_unchanged(self):
        response = _response("semantic")
        response["final_answer"] = "ไม่พบข้อมูลนี้ในเล่มหลักสูตร"
        with self._patch_runtime(), patch.object(
            ask, "answer_question_once", return_value=response
        ), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(ask.main(["question"]), 0)
        self.assertIn(response["final_answer"], output.getvalue())

    def test_blocked_no_data_displays_exact_fallback(self):
        response = {
            "route": None,
            "result": {
                "status": "no_data",
                "action": "no_data",
                "blocking_ambiguity": (),
                "context_conflicts": (),
                "resolved_program": None,
                "resolved_plans": (),
                "course_references": [],
            },
        }
        with self._patch_runtime(response), patch.object(
            ask, "answer_question_once", return_value=response
        ), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(ask.main(["06019999 ชื่ออะไร"]), 0)

        self.assertIn("ตอบ: ไม่พบข้อมูลนี้ในเล่มหลักสูตร", output.getvalue())
        self.assertEqual(response["result"]["status"], "no_data")

    def test_internal_provenance_debug_lines_are_not_shown_in_answer_text(self):
        response = _response("semantic")
        response["final_answer"] = (
            "คำตอบที่ยืนยันได้\n"
            '"source_page": 12\n'
            "source_file: it_page_012.png\n"
            "provenance: internal"
        )
        with self._patch_runtime(), patch.object(
            ask, "answer_question_once", return_value=response
        ), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(ask.main(["question"]), 0)

        displayed = output.getvalue()
        self.assertIn("ตอบ: คำตอบที่ยืนยันได้", displayed)
        self.assertNotIn("source_page:", displayed)
        self.assertNotIn("source_file:", displayed)
        self.assertNotIn("provenance:", displayed)

    def test_sources_group_pages_by_book_sorted_without_duplicates(self):
        result = {
            "provenance": [
                {"program": "IT", "source_filename": "it_page_039.png", "source_page": 39},
                {"program": "IT", "source_filename": "it_page_032.png", "source_page": 32},
                {"program": "IT", "source_filename": "it_page_328.png", "source_page": 328},
                {"program": "IT", "source_filename": "it_page_032.png", "source_page": 32},
            ]
        }

        self.assertEqual(
            ask._format_sources({"route": None, "result": result}),
            "เล่มหลักสูตร: IT\nหน้า: 32, 39, 328",
        )

    def test_sources_group_multiple_books_in_separate_blocks(self):
        result = {
            "provenance": [
                {"program": "IT", "source_filename": "it_page_032.png", "source_page": 32},
                {"program": "DSBA", "source_filename": "dsba_page_029.png", "source_page": 29},
                {"program": "IT", "source_filename": "it_page_328.png", "source_page": 328},
                {"program": "DSBA", "source_filename": "dsba_page_324.png", "source_page": 324},
            ]
        }

        formatted = ask._format_sources({"route": None, "result": result})

        self.assertEqual(
            formatted,
            "เล่มหลักสูตร: IT\nหน้า: 32, 328\n\nเล่มหลักสูตร: DSBA\nหน้า: 29, 324",
        )

    def test_sources_never_expose_internal_filenames_or_metadata(self):
        result = {
            "provenance": [
                {
                    "program": "IT",
                    "source_filename": "it_page_039.png",
                    "source_page": 39,
                    "source_uri": "repo:outputs/llm/x.json",
                    "source_locator": "locator-1",
                    "provenance_id": 182,
                }
            ]
        }

        formatted = ask._format_sources({"route": None, "result": result})

        for leaked in (
            "it_page_039.png",
            ".png",
            "source_uri",
            "source_locator",
            "provenance_id",
            "repo:outputs",
        ):
            self.assertNotIn(leaked, formatted)
        self.assertIn("เล่มหลักสูตร: IT", formatted)
        self.assertIn("หน้า: 39", formatted)

    def test_sources_leave_underlying_provenance_unchanged(self):
        import copy

        provenance = [
            {"program": "IT", "source_filename": "it_page_039.png", "source_page": [39, 32]},
            {"program": "DSBA", "source_filename": "dsba_page_029.png", "source_page": 29},
        ]
        before = copy.deepcopy(provenance)

        ask._format_sources({"route": None, "result": {"provenance": provenance}})

        self.assertEqual(provenance, before)

    def test_sources_without_pages_fail_gracefully(self):
        self.assertEqual(
            ask._format_sources({"route": None, "result": {"provenance": []}}),
            "ไม่พบ provenance ในผลลัพธ์",
        )
        self.assertEqual(
            ask._format_sources(
                {
                    "route": None,
                    "result": {
                        "provenance": [
                            {"program": "IT", "source_filename": "it_page_039.png"}
                        ]
                    },
                }
            ),
            "เล่มหลักสูตร: IT\nหน้า: ไม่ระบุ",
        )

    def test_no_data_source_output_remains_unchanged(self):
        response = {
            "route": None,
            "result": {
                "status": "no_data",
                "action": "no_data",
                "blocking_ambiguity": (),
                "context_conflicts": (),
                "resolved_program": None,
                "resolved_plans": (),
                "course_references": [],
            },
        }
        output = io.StringIO()
        with redirect_stdout(output):
            ask._print_result("question", response, show_question=False)

        self.assertIn("ตอบ: ไม่พบข้อมูลนี้ในเล่มหลักสูตร", output.getvalue())

    def test_ask_does_not_rebuild_index(self):
        response = _response()
        with self._patch_runtime(response), patch.object(
            ask, "answer_question_once", return_value=response
        ), patch("rag.hybrid_demo.ensure_index") as ensure:
            self.assertEqual(ask.main(["question"]), 0)
        ensure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
