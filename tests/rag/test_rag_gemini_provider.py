import os
import unittest
from unittest.mock import patch

from rag.providers.gemini import DEFAULT_MODEL, make_gemini_callable


class _Response:
    text = "SELECT course_code FROM courses"


class _Models:
    def __init__(self):
        self.prompts = []

    def generate_content(self, *, model, contents, config=None):
        self.prompts.append((model, contents, config))
        return _Response()


class _SequenceModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append((model, contents, config))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


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
            [call[:2] for call in client.models.prompts],
            [(DEFAULT_MODEL, "return a read-only query")],
        )

    def test_generation_uses_zero_temperature_and_preserves_prompt_model_and_call_count(self):
        client = _Client()
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("rag.providers.gemini._create_client", return_value=client),
        ):
            result = make_gemini_callable()("original prompt")

        self.assertEqual(result, _Response.text)
        self.assertEqual(len(client.models.prompts), 1)
        model, prompt, config = client.models.prompts[0]
        self.assertEqual(model, DEFAULT_MODEL)
        self.assertEqual(prompt, "original prompt")
        self.assertEqual(config.temperature, 0.0)
        self.assertIsNone(config.response_mime_type)
        self.assertIsNone(config.response_schema)

    def test_json_schema_configuration_is_opt_in_and_reaches_sdk(self):
        client = _Client()
        schema = {
            "type": "object",
            "properties": {"task_type": {"type": "string", "enum": ["none"]}},
            "required": ["task_type"],
            "additionalProperties": False,
        }
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("rag.providers.gemini._create_client", return_value=client),
        ):
            model_callable = make_gemini_callable()
            result = model_callable(
                "hard interpreter prompt",
                response_mime_type="application/json",
                response_json_schema=schema,
            )
            model_callable("ordinary SQL prompt")

        self.assertEqual(result, _Response.text)
        self.assertEqual(len(client.models.prompts), 2)
        model, prompt, config = client.models.prompts[0]
        self.assertEqual((model, prompt), (DEFAULT_MODEL, "hard interpreter prompt"))
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.response_json_schema, schema)
        self.assertIsNone(config.response_schema)
        ordinary_config = client.models.prompts[1][2]
        self.assertEqual(client.models.prompts[1][1], "ordinary SQL prompt")
        self.assertIsNone(ordinary_config.response_mime_type)
        self.assertIsNone(ordinary_config.response_schema)
        self.assertIsNone(ordinary_config.response_json_schema)

    def test_response_schema_and_response_json_schema_remain_distinct(self):
        client = _Client()
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("rag.providers.gemini._create_client", return_value=client),
        ):
            make_gemini_callable()(
                "legacy schema call",
                response_mime_type="application/json",
                response_schema=schema,
            )

        config = client.models.prompts[0][2]
        self.assertIsNotNone(config.response_schema)
        self.assertIsNone(config.response_json_schema)

    def test_transient_connection_failure_retries_once_then_succeeds(self):
        client = _Client()
        client.models = _SequenceModels(
            [ConnectionError("temporary disconnect"), _Response()]
        )
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("rag.providers.gemini._create_client", return_value=client),
            patch("rag.providers.gemini.time.sleep") as sleep,
        ):
            result = make_gemini_callable()("question")

        self.assertEqual(result, _Response.text)
        self.assertEqual(len(client.models.calls), 2)
        sleep.assert_called_once()
        self.assertGreaterEqual(sleep.call_args.args[0], 0)
        self.assertLessEqual(sleep.call_args.args[0], 0.5)

    def test_timeout_retries_once_then_succeeds(self):
        client = _Client()
        client.models = _SequenceModels([TimeoutError("slow request"), _Response()])
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("rag.providers.gemini._create_client", return_value=client),
            patch("rag.providers.gemini.time.sleep") as sleep,
        ):
            result = make_gemini_callable()("question")

        self.assertEqual(result, _Response.text)
        self.assertEqual(len(client.models.calls), 2)
        sleep.assert_called_once()
        self.assertGreaterEqual(sleep.call_args.args[0], 0)
        self.assertLessEqual(sleep.call_args.args[0], 0.5)

    def test_retryable_http_status_retries_once(self):
        from google.genai.errors import ClientError, ServerError

        for transient_error in (
            ClientError(429, {"message": "resource exhausted"}),
            ServerError(503, {"message": "temporarily unavailable"}),
        ):
            with self.subTest(status=transient_error.code):
                client = _Client()
                client.models = _SequenceModels([transient_error, _Response()])
                with (
                    patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                    patch("rag.providers.gemini._create_client", return_value=client),
                    patch("rag.providers.gemini.time.sleep") as sleep,
                ):
                    self.assertEqual(
                        make_gemini_callable()("question"), _Response.text
                    )
                self.assertEqual(len(client.models.calls), 2)
                sleep.assert_called_once()

    def test_api_client_error_logs_sanitized_details_for_structured_request(self):
        from google.genai.errors import ClientError

        secret = "test-api-key-must-not-appear"
        prompt = "private hard interpreter prompt"
        message = f"Invalid response schema\nrequest={prompt}\ncredential={secret}"
        error = ClientError(400, {"message": message})
        client = _Client()
        client.models = _SequenceModels([error])
        schema = {"type": "OBJECT", "properties": {}, "additionalProperties": False}

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": secret}),
            patch("rag.providers.gemini._create_client", return_value=client),
            patch("rag.providers.gemini.time.sleep") as sleep,
            self.assertLogs("rag.providers.gemini", level="WARNING") as logs,
        ):
            with self.assertRaises(ClientError):
                make_gemini_callable()(
                    prompt,
                    response_mime_type="application/json",
                    response_json_schema=schema,
                )

        self.assertEqual(len(client.models.calls), 1)
        sleep.assert_not_called()
        record = logs.records[0]
        self.assertEqual(record.api_code, 400)
        self.assertIn("Invalid response schema", record.api_message)
        self.assertNotIn("\n", record.api_message)
        self.assertTrue(record.structured_output_requested)
        self.assertEqual(record.exception_type, "ClientError")
        self.assertFalse(record.transient)
        self.assertEqual(record.stage, "generate_content")
        self.assertEqual(record.attempt, 1)
        self.assertLessEqual(len(record.api_message), 500)
        log_text = "\n".join(logs.output)
        self.assertNotIn(prompt, log_text)
        self.assertNotIn(secret, log_text)

    def test_api_client_error_logs_unstructured_flag_and_bounds_long_message(self):
        from google.genai.errors import ClientError

        client = _Client()
        client.models = _SequenceModels([ClientError(400, {"message": "x" * 900})])
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("rag.providers.gemini._create_client", return_value=client),
            self.assertLogs("rag.providers.gemini", level="WARNING") as logs,
        ):
            with self.assertRaises(ClientError):
                make_gemini_callable()("ordinary model prompt")

        self.assertEqual(len(client.models.calls), 1)
        self.assertFalse(logs.records[0].structured_output_requested)
        self.assertEqual(logs.records[0].api_code, 400)
        self.assertEqual(len(logs.records[0].api_message), 500)

    def test_api_error_without_code_or_message_attributes_logs_nulls(self):
        from google.genai.errors import ClientError

        error = ClientError(400, {"message": "temporary diagnostic value"})
        del error.code
        del error.message
        client = _Client()
        client.models = _SequenceModels([error])
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("rag.providers.gemini._create_client", return_value=client),
            self.assertLogs("rag.providers.gemini", level="WARNING") as logs,
        ):
            with self.assertRaises(ClientError):
                make_gemini_callable()("prompt")

        self.assertIsNone(logs.records[0].api_code)
        self.assertIsNone(logs.records[0].api_message)
        self.assertEqual(len(client.models.calls), 1)

    def test_transient_failure_twice_stops_after_two_attempts_and_logs_safely(self):
        secret = "test-api-key-must-not-appear"
        client = _Client()
        client.models = _SequenceModels(
            [ConnectionError(secret), ConnectionError(secret)]
        )
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": secret}),
            patch("rag.providers.gemini._create_client", return_value=client),
            patch("rag.providers.gemini.time.sleep") as sleep,
            self.assertLogs("rag.providers.gemini", level="WARNING") as logs,
        ):
            with self.assertRaises(ConnectionError):
                make_gemini_callable()("question")

        self.assertEqual(len(client.models.calls), 2)
        sleep.assert_called_once()
        log_text = "\n".join(logs.output)
        self.assertIn("attempt=1/2", log_text)
        self.assertIn("attempt=2/2", log_text)
        self.assertIn("ConnectionError", log_text)
        self.assertIn("transient=true", log_text)
        self.assertNotIn(secret, log_text)

    def test_authentication_bad_request_and_model_not_found_are_not_retried(self):
        from google.genai.errors import ClientError

        for error in (
            ClientError(401, {"message": "unauthorized"}),
            ClientError(400, {"message": "bad request"}),
            ClientError(404, {"message": "model not found"}),
        ):
            with self.subTest(status=error.code):
                client = _Client()
                client.models = _SequenceModels([error, _Response()])
                with (
                    patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                    patch("rag.providers.gemini._create_client", return_value=client),
                    patch("rag.providers.gemini.time.sleep") as sleep,
                ):
                    with self.assertRaises(ClientError):
                        make_gemini_callable()("question")
                self.assertEqual(len(client.models.calls), 1)
                sleep.assert_not_called()

    def test_client_uses_bounded_timeout_without_sdk_retries(self):
        from google import genai

        with patch.object(genai, "Client") as client_factory:
            from rag.providers.gemini import _create_client

            _create_client("test-key")

        options = client_factory.call_args.kwargs["http_options"]
        self.assertEqual(options.timeout, 20_000)
        self.assertEqual(options.retry_options.attempts, 1)

    def test_factory_fails_clearly_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                make_gemini_callable()


if __name__ == "__main__":
    unittest.main()
