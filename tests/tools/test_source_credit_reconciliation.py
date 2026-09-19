import unittest

from src.pipeline.tools.extraction.tool import _reconcile_source_backed_credit


class SourceCreditReconciliationTests(unittest.TestCase):
    def _course(self, code="C001", credits="(0-2-1)"):
        return {
            "code": code,
            "name_th": "ชื่อวิชา",
            "credits": credits,
            "year": 2,
            "semester": 1,
        }

    def _lookup(self, program="IT", plan="coop", code="C001", credits="1(0-2-1)"):
        return {(program, plan, code): {"code": code, "credits": credits}}

    def _reconcile(self, course, lookup):
        return _reconcile_source_backed_credit(
            course,
            lookup,
            program="IT",
            plan="coop",
        )

    def _source_course(
        self,
        *,
        program="IT",
        code="06016454",
        credits="",
        source_filename="it_page_354.png",
        source_page=354,
    ):
        course = self._course(code=code, credits=credits)
        course["source_provenance"] = [
            {
                "program": program,
                "source_filename": source_filename,
                "source_page": source_page,
                "document_category": "description",
            }
        ]
        return course

    def test_reconciles_parenthetical_credit_with_authoritative_unit(self):
        result = self._reconcile(
            self._course(),
            self._lookup(credits="1(0-2-1)"),
        )

        self.assertEqual(result["credits"], "1(0-2-1)")
        self.assertEqual(result["code"], "C001")
        self.assertEqual(result["name_th"], "ชื่อวิชา")

    def test_preserves_authoritative_zero_unit(self):
        result = self._reconcile(
            self._course(credits="(0-0-45)"),
            self._lookup(credits="0(0-0-45)"),
        )

        self.assertEqual(result["credits"], "0(0-0-45)")

    def test_reconciles_arbitrary_numeric_unit(self):
        result = self._reconcile(
            self._course(credits="(3-0-6)"),
            self._lookup(credits="6(3-0-6)"),
        )

        self.assertEqual(result["credits"], "6(3-0-6)")

    def test_missing_source_record_is_unchanged(self):
        course = self._course()

        result = self._reconcile(course, {})

        self.assertIs(result, course)
        self.assertEqual(result["credits"], "(0-2-1)")

    def test_tuple_mismatch_is_unchanged(self):
        course = self._course()

        result = self._reconcile(
            course,
            self._lookup(credits="1(0-3-1)"),
        )

        self.assertIs(result, course)

    def test_program_mismatch_is_unchanged(self):
        course = self._course()

        result = _reconcile_source_backed_credit(
            course,
            self._lookup(program="DSBA"),
            program="IT",
            plan="coop",
        )

        self.assertIs(result, course)

    def test_plan_mismatch_is_unchanged(self):
        course = self._course()

        result = _reconcile_source_backed_credit(
            course,
            self._lookup(plan="no_coop"),
            program="IT",
            plan="coop",
        )

        self.assertIs(result, course)

    def test_course_code_mismatch_is_unchanged(self):
        course = self._course(code="C002")

        result = self._reconcile(course, self._lookup(code="C001"))

        self.assertIs(result, course)

    def test_already_correct_credit_is_unchanged(self):
        course = self._course(credits="1(0-2-1)")

        result = self._reconcile(
            course,
            self._lookup(credits="1(0-2-1)"),
        )

        self.assertIs(result, course)

    def test_no_leading_unit_is_fabricated_without_authoritative_full_credit(self):
        course = self._course()

        result = self._reconcile(
            course,
            self._lookup(credits="(0-2-1)"),
        )

        self.assertIs(result, course)
        self.assertEqual(result["credits"], "(0-2-1)")

    def test_source_verified_it_description_credit_repair_requires_exact_identity(self):
        course = self._source_course()
        result = self._reconcile(course, self._lookup(code="06016454", credits="3(3-0-6)"))

        self.assertEqual(result["credits"], "3(3-0-6)")
        self.assertEqual(result["source_provenance"], course["source_provenance"])

    def test_source_verified_bit_description_credit_repair_requires_exact_identity(self):
        course = self._source_course(
            program="BIT",
            code="06036135",
            source_filename="bit_page_252_ocr.json",
            source_page=252,
        )
        result = _reconcile_source_backed_credit(
            course,
            self._lookup(program="BIT", code="06036135", credits="3(3-0-6)"),
            program="BIT",
            plan="coop",
        )

        self.assertEqual(result["credits"], "3(3-0-6)")

    def test_source_verified_repair_rejects_wrong_page_filename_program_or_course(self):
        lookup = self._lookup(code="06016454", credits="3(3-0-6)")
        for kwargs in (
            {"source_page": 355},
            {"source_filename": "it_page_355.png"},
            {"program": "BIT"},
            {"code": "06016455"},
        ):
            course = self._source_course(**kwargs)
            result = self._reconcile(course, lookup)
            self.assertIs(result, course)
            self.assertEqual(result["credits"], "")

    def test_source_verified_repair_preserves_already_correct_credit(self):
        course = self._source_course(credits="3(3-0-6)")
        result = self._reconcile(course, self._lookup(code="06016454", credits="3(3-0-6)"))

        self.assertIs(result, course)

    def test_conflicting_valid_credit_fails_closed(self):
        course = self._source_course(credits="4(3-0-6)")
        result = self._reconcile(course, self._lookup(code="06016454", credits="3(3-0-6)"))

        self.assertIs(result, course)
        self.assertEqual(result["credits"], "4(3-0-6)")

    def test_generic_malformed_credit_gets_no_guessed_value(self):
        course = self._source_course(code="C001", credits="3(2-2")
        result = self._reconcile(course, self._lookup(code="06016454", credits="3(3-0-6)"))

        self.assertIs(result, course)
        self.assertEqual(result["credits"], "3(2-2")


if __name__ == "__main__":
    unittest.main()
