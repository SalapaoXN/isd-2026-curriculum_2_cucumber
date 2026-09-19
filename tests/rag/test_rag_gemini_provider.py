import os
import unittest
from unittest.mock import patch

from rag.providers.gemini import DEFAULT_MODEL, make_gemini_callable


class _Response:
    text = "SELECT course_code FROM courses"


class _Models:
    def __init__(self):
        self.prompts = []

    def generate_content(self, *, model, contents):
        self.prompts.append((model, contents))
        return _Response()


class _Client:
    def __init__(self):
        self.models = _Models()


class GeminiProviderTest(unittest.TestCase):
    def test_default_model_is_gemini_3_5_flash_lite(self):
        self.assertEqual(DEFAULT_MODEL, "gemini-3.5-flash-lite")

    def test_factory_validates_key_and_creates_mocked_client_lazily(self):
        client = _Client()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch(
            "rag.providers.gemini._create_client", return_value=client
        ) as create_client:
            model_callable = make_gemini_callable()
            create_client.assert_not_called()

            result = model_callable("return a read-only query")

        create_client.assert_called_once_with("test-key")
        self.assertEqual(result, _Response.text)
        self.assertEqual(
            client.models.prompts,
            [(DEFAULT_MODEL, "return a read-only query")],
        )

    def test_factory_fails_clearly_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                make_gemini_callable()


if __name__ == "__main__":
    unittest.main()
