import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rag.demo import run_demo


class RagDemoTest(unittest.TestCase):
    def test_uses_the_canonical_unified_database(self):
        sources = [Path("outputs/consolidated/it/coop/full/merged_it_coop_full.json")]
        evidence = [
            {
                "chunk_id": "chunk-1",
                "distance": 0.1,
                "text": "curriculum evidence",
                "source_page": [12],
            }
        ]

        with patch("rag.demo.canonical_source_paths", return_value=sources), patch(
            "rag.demo.query_index", return_value=evidence
        ) as query_index, redirect_stdout(io.StringIO()):
            run_demo("วิชาฐานข้อมูล", top_k=1)

        query_index.assert_called_once_with(sources, "วิชาฐานข้อมูล", top_k=1)


if __name__ == "__main__":
    unittest.main()
