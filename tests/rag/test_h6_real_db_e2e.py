import unittest
from pathlib import Path

from rag.qa import ask


DB_PATH = Path(__file__).parents[2] / "cucumber_outputs" / "runtime" / "curriculum.db"


class RealRuntimeQaRegressionTests(unittest.TestCase):
    def test_exact_credit_is_canonical_and_provenanced(self):
        result = ask(
            DB_PATH,
            "IT วิชา 06016454 มีกี่หน่วยกิต",
            structured_model_callable=None,
            answer_model_callable=None,
        )["result"]

        self.assertEqual(result.status, "answer")
        self.assertIn("3", result.final_answer)
        self.assertTrue(result.provenance)
        self.assertTrue(any(claim.operation == "sum_credits" for claim in result.claims))

    def test_placement_answer_retains_canonical_provenance(self):
        result = ask(
            DB_PATH,
            "IT 06016454 อยู่ปีไหน เทอมไหน",
            structured_model_callable=None,
            answer_model_callable=None,
        )["result"]

        self.assertEqual(result.status, "answer")
        self.assertTrue(result.provenance)
        self.assertTrue(any(claim.operation == "placement" for claim in result.claims))

    def test_unsupported_request_fails_closed_without_evidence(self):
        result = ask(
            DB_PATH,
            "IT 06016454 เรียนยากไหม",
            structured_model_callable=None,
            answer_model_callable=None,
        )["result"]

        self.assertEqual(result["status"], "unsupported")


if __name__ == "__main__":
    unittest.main()
