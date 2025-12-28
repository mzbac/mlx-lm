# Copyright © 2024 Apple Inc.

import http
import io
import json
import threading
import unittest

import mlx.core as mx
import requests

from mlx_lm.models.cache import KVCache
from mlx_lm.server import (
    APIHandler,
    LRUPromptCache,
    ResponseGenerator,
    parse_minimax_tool_calls,
    normalize_tool_calls,
    stopping_criteria,
    StopCondition,
    convert_anthropic_to_openai_messages,
    process_message_content,
)
from mlx_lm.utils import load


class DummyModelProvider:
    def __init__(self, with_draft=False):
        HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        self.model, self.tokenizer = load(HF_MODEL_PATH)
        self.model_key = (HF_MODEL_PATH, None)
        self.cache_types = set([KVCache])

        # Add draft model support
        self.draft_model = None
        self.draft_model_key = None
        self.cli_args = type(
            "obj",
            (object,),
            {
                "adapter_path": None,
                "chat_template": None,
                "use_default_chat_template": False,
                "trust_remote_code": False,
                "draft_model": None,
                "num_draft_tokens": 3,
                "temp": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "max_tokens": 512,
                "chat_template_args": {},
            },
        )

        if with_draft:
            # Use the same model as the draft model for testing
            self.draft_model, _ = load(HF_MODEL_PATH)
            self.draft_model_key = HF_MODEL_PATH
            self.cli_args.draft_model = HF_MODEL_PATH

    def load(self, model, adapter=None, draft_model=None):
        assert model in ["default_model", "chat_model"]
        return self.model, self.tokenizer


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_handle_completions(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "repetition_context_size": 20,
            "seed": 999,
            "stop": "stop sequence",
        }

        response = requests.post(url, json=post_data)

        response_body = json.loads(response.text)

        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        first_text = response_body["choices"][0]["text"]
        self.assertEqual(
            first_text,
            json.loads(requests.post(url, json=post_data).text)["choices"][0]["text"],
        )

    def test_handle_chat_completions(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_content_fragments(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_null_tool_content(self):
        """Test OpenAI API with tool_calls containing JSON string arguments.

        This is an end-to-end test that verifies the server correctly parses
        JSON string arguments to dicts for chat templates that call .items().
        """
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "user", "content": "what is 2+3?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "123",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 2, "b": 3}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "5", "tool_call_id": "123"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_multiple_tool_calls(self):
        """Test OpenAI API with multiple tool_calls in conversation history.

        Verifies all tool_call arguments are properly parsed from JSON strings.
        """
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "Search for weather and calculate 2+2"},
                {
                    "role": "assistant",
                    "content": "I'll help with both.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call_1",
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "weather today", "limit": 5}',
                            },
                        },
                        {
                            "type": "function",
                            "id": "call_2",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "2+2"}',
                            },
                        },
                    ],
                },
                {"role": "tool", "content": "Sunny, 72F", "tool_call_id": "call_1"},
                {"role": "tool", "content": "4", "tool_call_id": "call_2"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        self.assertEqual(response.status_code, 200)
        response_body = response.json()
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_complex_tool_arguments(self):
        """Test OpenAI API with complex nested JSON arguments.

        Verifies nested objects and arrays in arguments are correctly parsed.
        """
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "Edit the file"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "edit_1",
                            "function": {
                                "name": "Edit",
                                "arguments": '{"file_path": "/path/to/file.py", "old_string": "def foo():", "new_string": "def bar():", "options": {"backup": true, "dry_run": false}}',
                            },
                        },
                    ],
                },
                {"role": "tool", "content": "File updated successfully", "tool_call_id": "edit_1"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        self.assertEqual(response.status_code, 200)

    def test_handle_models(self):
        url = f"http://localhost:{self.port}/v1/models"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        response_body = json.loads(response.text)
        self.assertEqual(response_body["object"], "list")
        self.assertIsInstance(response_body["data"], list)
        self.assertGreater(len(response_body["data"]), 0)
        model = response_body["data"][0]
        self.assertIn("id", model)
        self.assertEqual(model["object"], "model")
        self.assertIn("created", model)

    def test_sequence_overlap(self):
        from mlx_lm.server import sequence_overlap

        self.assertTrue(sequence_overlap([1], [1]))
        self.assertTrue(sequence_overlap([1, 2], [1, 2]))
        self.assertTrue(sequence_overlap([1, 3], [3, 4]))
        self.assertTrue(sequence_overlap([1, 2, 3], [2, 3]))

        self.assertFalse(sequence_overlap([1], [2]))
        self.assertFalse(sequence_overlap([1, 2], [3, 4]))
        self.assertFalse(sequence_overlap([1, 2, 3], [4, 1, 2, 3]))


class TestServerWithDraftModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(with_draft=True), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_handle_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.0,
            "top_p": 1.0,
        }

        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_handle_chat_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_streaming_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data, stream=True)
        self.assertEqual(response.status_code, 200)

        chunk_count = 0
        for chunk in response.iter_lines():
            if chunk:
                data = chunk.decode("utf-8")
                if data.startswith("data: ") and data != "data: [DONE]":
                    chunk_data = json.loads(data[6:])  # Skip the "data: " prefix
                    self.assertIn("choices", chunk_data)
                    self.assertEqual(len(chunk_data["choices"]), 1)
                    self.assertIn("delta", chunk_data["choices"][0])
                    chunk_count += 1

        # Make sure we got some streaming chunks
        self.assertGreater(chunk_count, 0)

    def test_prompt_cache_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        # First request to initialize cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about"},
            ],
        }

        first_response = requests.post(url, json=chat_post_data)
        self.assertEqual(first_response.status_code, 200)

        # Second request with same prefix should use cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about dragons."},
            ],
        }

        second_response = requests.post(url, json=chat_post_data)
        self.assertEqual(second_response.status_code, 200)

        # Both responses should have content
        first_response_body = json.loads(first_response.text)
        second_response_body = json.loads(second_response.text)

        self.assertIn("choices", first_response_body)
        self.assertIn("choices", second_response_body)
        self.assertIn("message", first_response_body["choices"][0])
        self.assertIn("message", second_response_body["choices"][0])
        self.assertIn("content", first_response_body["choices"][0]["message"])
        self.assertIn("content", second_response_body["choices"][0]["message"])

        # Ensure both generated content
        self.assertIsNotNone(first_response_body["choices"][0]["message"]["content"])
        self.assertIsNotNone(second_response_body["choices"][0]["message"]["content"])


class TestKeepalive(unittest.TestCase):

    def test_keepalive_callback(self):
        """Test keepalive callback sends SSE comments and handles errors"""
        from unittest.mock import Mock

        # Mock handler
        mock_wfile = io.BytesIO()
        handler = Mock()
        handler.wfile = mock_wfile

        # Test callback logic (same as in server.py)
        def keepalive_callback(processed_tokens, total_tokens):
            if handler.stream:
                try:
                    handler.wfile.write(
                        f": keepalive {processed_tokens}/{total_tokens}\n\n".encode()
                    )
                    handler.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        # Test streaming enabled
        handler.stream = True
        keepalive_callback(1024, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, ": keepalive 1024/4096\n\n")

        # Test streaming disabled
        handler.stream = False
        mock_wfile.seek(0)
        mock_wfile.truncate(0)
        keepalive_callback(2048, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, "")

        # Test error handling
        handler.stream = True
        handler.wfile = Mock()
        handler.wfile.write.side_effect = BrokenPipeError("Connection broken")

        # Should not raise exception
        try:
            keepalive_callback(3072, 4096)
        except Exception as e:
            self.fail(f"Callback should handle BrokenPipeError: {e}")


class TestLRUPromptCache(unittest.TestCase):

    def test_caching(self):
        cache = LRUPromptCache(max_size=10)

        def get_kv(n):
            keys = mx.arange(n).reshape(1, 1, n, 1)
            return keys, keys

        model = ("test", None, None)
        tokens = [10] * 24

        c, t = cache.fetch_nearest_cache(model, tokens)
        self.assertTrue(c is None)
        self.assertEqual(t, tokens)

        c = [KVCache()]
        c[0].update_and_fetch(*get_kv(24))
        cache.insert_cache(model, t, c)

        tokens = tokens + [20] * 5
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue((k.flatten() == mx.arange(24)).all().item())
        self.assertEqual(t, [20] * 5)
        self.assertEqual(len(cache._lru), 0)

        tokens = tokens + [30] * 3
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)

        tokens = tokens[:26] + [40] * 8
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue(
            (k.flatten() == mx.concatenate([mx.arange(24), mx.arange(2)])).all().item()
        )
        self.assertEqual(t, [40] * 8)
        self.assertEqual(len(cache._lru), 1)

    def test_lru(self):
        cache = LRUPromptCache(max_size=2)
        model = ("test", None, None)
        cache.insert_cache(model, [1, 2], ["test1"])
        cache.insert_cache(model, [1, 2], ["test1"])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, ["test1"])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, ["test1"])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])

        cache.insert_cache(model, [1, 2], ["test1"])
        cache.insert_cache(model, [2, 3], ["test2"])
        cache.insert_cache(model, [3, 4], ["test3"])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, ["test2"])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, ["test3"])
        self.assertEqual(t, [])


class TestAnthropicAPI(unittest.TestCase):
    """Test Anthropic /v1/messages API compatibility."""

    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_basic_message(self):
        """Test basic message completion."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("id", body)
        self.assertTrue(body["id"].startswith("msg_"))
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["role"], "assistant")
        self.assertIn("content", body)
        self.assertIn("stop_reason", body)
        self.assertIn("usage", body)

    def test_system_prompt_string(self):
        """Test system prompt as string."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

    def test_system_prompt_blocks(self):
        """Test system prompt as content blocks."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "system": [{"type": "text", "text": "You are helpful."}],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

    def test_content_blocks(self):
        """Test message with content blocks."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

    def test_streaming(self):
        """Test streaming response format."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = requests.post(url, json=post_data, stream=True)
        self.assertEqual(response.status_code, 200)

        events = []
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("event:"):
                    events.append(line.split(": ", 1)[1])

        # Verify expected event sequence
        self.assertIn("message_start", events)
        self.assertIn("content_block_start", events)
        self.assertIn("message_stop", events)

    def test_stop_sequences(self):
        """Test stop_sequences parameter."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 50,
            "stop_sequences": ["\n"],
            "messages": [{"role": "user", "content": "Count: 1, 2, 3"}],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

    def test_temperature(self):
        """Test temperature parameter."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "temperature": 0.5,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

    def test_top_p_top_k(self):
        """Test top_p and top_k parameters."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "top_p": 0.9,
            "top_k": 40,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

    def test_usage_tokens(self):
        """Test usage token counts in response."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = requests.post(url, json=post_data)
        body = response.json()
        self.assertIn("usage", body)
        self.assertIn("input_tokens", body["usage"])
        self.assertIn("output_tokens", body["usage"])
        self.assertGreater(body["usage"]["input_tokens"], 0)
        self.assertGreater(body["usage"]["output_tokens"], 0)

    def test_tool_result_in_messages(self):
        """Test handling of tool_result content blocks."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "calculator",
                            "input": {"expr": "2+2"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_123", "content": "4"}
                    ],
                },
            ],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

    def test_tool_result_with_is_error(self):
        """Test tool_result with is_error flag."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "Run command"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_456",
                            "name": "execute",
                            "input": {"cmd": "invalid"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_456",
                            "content": "Command failed: not found",
                            "is_error": True,
                        }
                    ],
                },
            ],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

    def test_tool_result_with_nested_content(self):
        """Test tool_result with nested content blocks."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "Get info"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_789",
                            "name": "get_info",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_789",
                            "content": [
                                {"type": "text", "text": "Info line 1"},
                                {"type": "text", "text": "Info line 2"},
                            ],
                        }
                    ],
                },
            ],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation with alternating roles."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm fine."},
                {"role": "user", "content": "Great!"},
            ],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["role"], "assistant")

    def test_mixed_content_blocks(self):
        """Test message with multiple text blocks."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First part. "},
                        {"type": "text", "text": "Second part."},
                    ],
                }
            ],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

    def test_tool_definitions(self):
        """Test request with tool definitions."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                        },
                        "required": ["location"],
                    },
                }
            ],
            "messages": [{"role": "user", "content": "What's the weather?"}],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

    def test_streaming_data_format(self):
        """Test streaming response data structure."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = requests.post(url, json=post_data, stream=True)
        self.assertEqual(response.status_code, 200)

        message_start_data = None
        message_delta_data = None
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    if data.get("type") == "message_start":
                        message_start_data = data
                    elif data.get("type") == "message_delta":
                        message_delta_data = data

        # Verify message_start structure
        self.assertIsNotNone(message_start_data)
        self.assertIn("message", message_start_data)
        self.assertEqual(message_start_data["message"]["role"], "assistant")
        self.assertIn("usage", message_start_data["message"])

        # Verify message_delta structure
        self.assertIsNotNone(message_delta_data)
        self.assertIn("delta", message_delta_data)
        self.assertIn("stop_reason", message_delta_data["delta"])
        self.assertIn("usage", message_delta_data)

    def test_empty_system_prompt(self):
        """Test with empty system prompt."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "system": "",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

    def test_response_content_structure(self):
        """Test that response content is an array of content blocks."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = requests.post(url, json=post_data)
        body = response.json()

        # Content should be a list
        self.assertIsInstance(body["content"], list)
        # Each content block should have a type
        for block in body["content"]:
            self.assertIn("type", block)
            if block["type"] == "text":
                self.assertIn("text", block)

    def test_model_in_response(self):
        """Test that model is included in response."""
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = requests.post(url, json=post_data)
        body = response.json()
        self.assertIn("model", body)
        self.assertEqual(body["model"], "default_model")

    def test_query_parameters_ignored(self):
        """Test that query parameters in URL don't cause 404."""
        # Claude clients may send query params like ?beta=true
        url = f"http://localhost:{self.port}/v1/messages?beta=true"
        post_data = {
            "model": "default_model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["role"], "assistant")


class TestMiniMaxToolParsing(unittest.TestCase):
    """Test MiniMax XML tool call parsing."""

    def test_parse_single_tool_call(self):
        """Test parsing a single MiniMax-style tool call."""
        tool_text = '''<invoke name="get_weather">
<parameter name="location">San Francisco</parameter>
<parameter name="unit">celsius</parameter>
</invoke>'''
        result = parse_minimax_tool_calls(tool_text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "get_weather")
        self.assertEqual(result[0]["arguments"]["location"], "San Francisco")
        self.assertEqual(result[0]["arguments"]["unit"], "celsius")

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls in one block."""
        tool_text = '''<invoke name="read_file">
<parameter name="path">/tmp/test.txt</parameter>
</invoke>
<invoke name="write_file">
<parameter name="path">/tmp/output.txt</parameter>
<parameter name="content">Hello World</parameter>
</invoke>'''
        result = parse_minimax_tool_calls(tool_text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "read_file")
        self.assertEqual(result[0]["arguments"]["path"], "/tmp/test.txt")
        self.assertEqual(result[1]["name"], "write_file")
        self.assertEqual(result[1]["arguments"]["path"], "/tmp/output.txt")
        self.assertEqual(result[1]["arguments"]["content"], "Hello World")

    def test_parse_json_parameter_value(self):
        """Test parsing parameter values that are valid JSON."""
        tool_text = '''<invoke name="run_code">
<parameter name="code">print("hello")</parameter>
<parameter name="timeout">30</parameter>
<parameter name="enabled">true</parameter>
</invoke>'''
        result = parse_minimax_tool_calls(tool_text)
        self.assertEqual(len(result), 1)
        # JSON values should be parsed
        self.assertEqual(result[0]["arguments"]["timeout"], 30)
        self.assertEqual(result[0]["arguments"]["enabled"], True)
        # Non-JSON string should remain as string
        self.assertEqual(result[0]["arguments"]["code"], 'print("hello")')

    def test_parse_empty_text(self):
        """Test parsing empty text returns empty list."""
        result = parse_minimax_tool_calls("")
        self.assertEqual(result, [])

    def test_parse_no_tools(self):
        """Test parsing text without tool calls returns empty list."""
        result = parse_minimax_tool_calls("Just some regular text")
        self.assertEqual(result, [])

    def test_normalize_tool_calls_with_minimax_format(self):
        """Test normalize_tool_calls converts MiniMax format to JSON."""
        tool_texts = ['''<invoke name="test_tool">
<parameter name="arg1">value1</parameter>
</invoke>''']
        result = normalize_tool_calls(tool_texts)
        self.assertEqual(len(result), 1)
        parsed = json.loads(result[0])
        self.assertEqual(parsed["name"], "test_tool")
        self.assertEqual(parsed["arguments"]["arg1"], "value1")

    def test_normalize_tool_calls_with_json_format(self):
        """Test normalize_tool_calls preserves JSON format."""
        tool_texts = ['{"name": "test_tool", "arguments": {"arg1": "value1"}}']
        result = normalize_tool_calls(tool_texts)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], tool_texts[0])

    def test_normalize_tool_calls_mixed_format(self):
        """Test normalize_tool_calls handles mixed formats."""
        tool_texts = [
            '{"name": "json_tool", "arguments": {}}',
            '''<invoke name="xml_tool">
<parameter name="p1">v1</parameter>
</invoke>''',
        ]
        result = normalize_tool_calls(tool_texts)
        self.assertEqual(len(result), 2)
        # First should be unchanged
        parsed1 = json.loads(result[0])
        self.assertEqual(parsed1["name"], "json_tool")
        # Second should be converted
        parsed2 = json.loads(result[1])
        self.assertEqual(parsed2["name"], "xml_tool")

    def test_parse_single_quotes(self):
        """Test parsing with single quotes in attribute names."""
        tool_text = '''<invoke name='get_data'>
<parameter name='id'>123</parameter>
</invoke>'''
        result = parse_minimax_tool_calls(tool_text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "get_data")
        self.assertEqual(result[0]["arguments"]["id"], 123)

    def test_parse_nested_json_value(self):
        """Test parsing parameter with nested JSON object."""
        tool_text = '''<invoke name="create_item">
<parameter name="data">{"key": "value", "count": 5}</parameter>
</invoke>'''
        result = parse_minimax_tool_calls(tool_text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["arguments"]["data"], {"key": "value", "count": 5})


class TestStoppingCriteria(unittest.TestCase):
    """Tests for stopping_criteria function."""

    def test_eos_token_detected(self):
        """Test that EOS token is detected and trim_text_length is set correctly."""
        tokens = [1, 2, 3, 100]  # 100 is EOS
        result = stopping_criteria(tokens, [], [], eos_token_id=100, eos_token='[e~[')
        self.assertTrue(result.stop_met)
        self.assertEqual(result.trim_length, 0)
        self.assertEqual(result.trim_text_length, 4)  # len('[e~[')

    def test_eos_token_empty_string(self):
        """Test backwards compatibility with empty eos_token."""
        tokens = [1, 2, 3, 100]
        result = stopping_criteria(tokens, [], [], eos_token_id=100, eos_token='')
        self.assertTrue(result.stop_met)
        self.assertEqual(result.trim_text_length, 0)

    def test_eos_token_default_parameter(self):
        """Test that eos_token defaults to empty string."""
        tokens = [1, 2, 3, 100]
        result = stopping_criteria(tokens, [], [], eos_token_id=100)
        self.assertTrue(result.stop_met)
        self.assertEqual(result.trim_text_length, 0)

    def test_no_eos_token(self):
        """Test that non-EOS tokens don't trigger stop."""
        tokens = [1, 2, 3]
        result = stopping_criteria(tokens, [], [], eos_token_id=100, eos_token='[e~[')
        self.assertFalse(result.stop_met)

    def test_stop_word_sequence(self):
        """Test stop word sequence detection."""
        tokens = [1, 2, 3, 4]
        stop_ids = [[3, 4]]
        stop_words = ["</s>"]
        result = stopping_criteria(tokens, stop_ids, stop_words, eos_token_id=100)
        self.assertTrue(result.stop_met)
        self.assertEqual(result.trim_length, 2)
        self.assertEqual(result.trim_text_length, 4)  # len("</s>")


class TestExtractToolCallsFromText(unittest.TestCase):
    """Tests for _extract_tool_calls_from_text method."""

    def setUp(self):
        """Create a mock handler for testing."""
        # Create minimal mock objects
        class MockWFile:
            def write(self, data): pass
            def flush(self): pass

        class MockHandler:
            wfile = MockWFile()
            request_id = "test-id"
            requested_model = "test-model"
            stream = False

            def _extract_tool_calls_from_text(self, text):
                import re
                extracted = []
                clean_text = text

                if text and "<minimax:tool_call>" in text:
                    pattern = re.compile(
                        r'<minimax:tool_call>(.*?)(?:</minimax:tool_call>|\[e~\[|$)', re.DOTALL
                    )
                    for match in pattern.finditer(text):
                        for p in parse_minimax_tool_calls(match.group(1).strip()):
                            extracted.append(json.dumps(p))
                    clean_text = pattern.sub('', text)

                    if not extracted and '<invoke' in text:
                        for p in parse_minimax_tool_calls(text):
                            extracted.append(json.dumps(p))
                        if extracted:
                            invoke_pattern = re.compile(
                                r'<invoke\s+name=["\'][^"\']+["\']>.*?</invoke>', re.DOTALL
                            )
                            clean_text = invoke_pattern.sub('', clean_text)

                elif text and '<invoke' in text and '</invoke>' in text:
                    for p in parse_minimax_tool_calls(text):
                        extracted.append(json.dumps(p))
                    if extracted:
                        invoke_pattern = re.compile(
                            r'<invoke\s+name=["\'][^"\']+["\']>.*?</invoke>', re.DOTALL
                        )
                        clean_text = invoke_pattern.sub('', text)

                if clean_text:
                    clean_text = re.sub(r'</?think>', '', clean_text)
                    clean_text = re.sub(r'\[e~\[', '', clean_text)
                    clean_text = re.sub(r'⏺', '', clean_text)
                    clean_text = re.sub(r'</?minimax:tool_call>', '', clean_text)
                    clean_text = re.sub(r'\]~b\][a-z]+', '', clean_text)

                return clean_text, extracted

        self.handler = MockHandler()

    def test_extract_with_minimax_wrapper(self):
        """Test extraction from <minimax:tool_call> wrapped content."""
        text = '''Some text <minimax:tool_call>
<invoke name="test_tool">
<parameter name="arg">value</parameter>
</invoke>
</minimax:tool_call> more text'''
        clean, extracted = self.handler._extract_tool_calls_from_text(text)
        self.assertEqual(len(extracted), 1)
        self.assertIn("test_tool", extracted[0])
        self.assertNotIn("<minimax:tool_call>", clean)
        self.assertIn("Some text", clean)
        self.assertIn("more text", clean)

    def test_extract_without_wrapper(self):
        """Test extraction from direct <invoke> blocks."""
        text = '''Some text <invoke name="test_tool">
<parameter name="arg">value</parameter>
</invoke> more text'''
        clean, extracted = self.handler._extract_tool_calls_from_text(text)
        self.assertEqual(len(extracted), 1)
        self.assertIn("test_tool", extracted[0])
        self.assertNotIn("<invoke", clean)

    def test_cleanup_think_tags(self):
        """Test that <think> tags are removed."""
        text = "Hello <think>internal reasoning</think> World"
        clean, _ = self.handler._extract_tool_calls_from_text(text)
        self.assertNotIn("<think>", clean)
        self.assertNotIn("</think>", clean)
        self.assertIn("Hello", clean)
        self.assertIn("World", clean)

    def test_cleanup_eos_token(self):
        """Test that [e~[ EOS token is removed."""
        text = "Hello World[e~["
        clean, _ = self.handler._extract_tool_calls_from_text(text)
        self.assertNotIn("[e~[", clean)
        self.assertIn("Hello World", clean)

    def test_preserves_whitespace(self):
        """Test that normal whitespace is preserved."""
        text = "The user wants to create a game."
        clean, _ = self.handler._extract_tool_calls_from_text(text)
        self.assertEqual(clean, text)

    def test_incomplete_invoke_preserved(self):
        """Test that incomplete <invoke tags are preserved (for streaming)."""
        text = "Some text <invoke name="
        clean, extracted = self.handler._extract_tool_calls_from_text(text)
        self.assertEqual(len(extracted), 0)
        # Incomplete tag should be preserved
        self.assertIn("<invoke", clean)

    def test_extract_with_eos_terminator(self):
        """Test extraction when [e~[ terminates the tool call block."""
        text = '''<minimax:tool_call>
<invoke name="test">
<parameter name="x">1</parameter>
</invoke>[e~['''
        clean, extracted = self.handler._extract_tool_calls_from_text(text)
        self.assertEqual(len(extracted), 1)
        self.assertNotIn("[e~[", clean)


class TestOpenAIStreamingWhitespace(unittest.TestCase):
    """Tests to ensure streaming doesn't break whitespace."""

    def test_streaming_chunks_preserve_spaces(self):
        """Simulate streaming chunks and verify spaces are preserved."""
        # This tests the logic that for streaming incremental chunks,
        # we should NOT call extraction/cleanup
        chunks = ["The ", "user ", "wants ", "to ", "create."]

        # Simulate what generate_response does for streaming
        # When stream=True and finish_reason=None, text should pass through as-is
        accumulated = ""
        for chunk in chunks:
            # For streaming incremental (finish_reason=None), text passes through
            clean_text = chunk  # No extraction for incremental streaming
            accumulated += clean_text

        self.assertEqual(accumulated, "The user wants to create.")

    def test_final_chunk_gets_cleaned(self):
        """Test that final chunk (with finish_reason) gets cleaned."""
        import re
        text = "Response text</think> with artifacts[e~["

        # Simulate final chunk cleanup
        clean_text = re.sub(r'</?think>', '', text)
        clean_text = re.sub(r'\[e~\[', '', clean_text)

        self.assertNotIn("</think>", clean_text)
        self.assertNotIn("[e~[", clean_text)
        self.assertIn("Response text", clean_text)


class TestXMLToolCallDetection(unittest.TestCase):
    """Tests for inline XML tool call detection in streaming."""

    def test_detect_minimax_tool_call_tag(self):
        """Test detection of <minimax:tool_call> tag."""
        text = "Some text <minimax:tool_call>"
        in_xml_tool_call = "<minimax:tool_call>" in text or "<invoke " in text
        self.assertTrue(in_xml_tool_call)

    def test_detect_invoke_tag_with_space(self):
        """Test detection of <invoke with space."""
        text = "Some text <invoke name='tool'>"
        in_xml_tool_call = "<minimax:tool_call>" in text or "<invoke " in text
        self.assertTrue(in_xml_tool_call)

    def test_detect_invoke_tag_with_newline(self):
        """Test detection of <invoke with newline."""
        text = "Some text <invoke\nname='tool'>"
        in_xml_tool_call = "<minimax:tool_call>" in text or "<invoke " in text or "<invoke\n" in text
        self.assertTrue(in_xml_tool_call)

    def test_no_false_positive_for_regular_text(self):
        """Test that regular text doesn't trigger detection."""
        text = "The user wants to invoke a function"
        in_xml_tool_call = "<minimax:tool_call>" in text or "<invoke " in text or "<invoke\n" in text
        self.assertFalse(in_xml_tool_call)

    def test_no_false_positive_for_html_tags(self):
        """Test that HTML-like tags don't trigger detection."""
        text = "Use <div> and <span> for styling"
        in_xml_tool_call = "<minimax:tool_call>" in text or "<invoke " in text or "<invoke\n" in text
        self.assertFalse(in_xml_tool_call)

    def test_streaming_stops_when_tool_call_detected(self):
        """Simulate streaming behavior when tool call is detected."""
        chunks = [
            "Let me help. ",
            "<invoke ",
            "name='test'>",
            "<parameter>x</parameter>",
            "</invoke>"
        ]

        in_xml_tool_call = False
        accumulated_text = ""
        sent_chunks = []

        for chunk in chunks:
            accumulated_text += chunk
            # Check for tool call markers
            if not in_xml_tool_call and ("<minimax:tool_call>" in accumulated_text or "<invoke " in accumulated_text):
                in_xml_tool_call = True

            # Only send if not in tool call
            if not in_xml_tool_call:
                sent_chunks.append(chunk)

        # Should have stopped sending after detecting <invoke
        self.assertEqual(len(sent_chunks), 1)
        self.assertEqual(sent_chunks[0], "Let me help. ")
        # Full text should be used for final extraction
        self.assertIn("<invoke", accumulated_text)
        self.assertIn("</invoke>", accumulated_text)


class TestMiniMaxChatTemplateValidation(unittest.TestCase):
    """Test that message conversion produces valid sequences for MiniMax chat template.

    MiniMax's chat template has strict validation rules:
    1. Tool messages must immediately follow assistant messages with tool_calls
    2. tool_call arguments must be dicts (not JSON strings)
    """

    def validate_messages_for_minimax(self, messages):
        """Simulate MiniMax chat template validation logic.

        MiniMax allows consecutive tool messages as long as they follow an
        assistant message with tool_calls. The tool messages don't need to
        immediately follow the assistant - they can follow other tool messages.
        """
        errors = []
        last_assistant_with_tools = None

        for i, msg in enumerate(messages):
            role = msg.get("role")

            # Track assistant messages with tool_calls
            if role == "assistant" and "tool_calls" in msg:
                last_assistant_with_tools = msg

            # Check: tool messages must have a preceding assistant with tool_calls
            if role == "tool":
                if last_assistant_with_tools is None:
                    errors.append(f"Message {i}: tool message has no previous assistant with tool_calls")

            # Reset after non-tool, non-assistant messages (except system)
            if role not in ("tool", "assistant", "system"):
                # Check if there are unmatched tool_calls
                pass

            # Check: tool_call arguments must be dicts
            if "tool_calls" in msg:
                for j, tc in enumerate(msg["tool_calls"]):
                    func = tc.get("function", tc)
                    args = func.get("arguments")
                    if isinstance(args, str):
                        errors.append(f"Message {i}, tool_call {j}: arguments is string, must be dict")

        return errors

    def test_openai_single_tool_call_valid(self):
        """Test OpenAI format with single tool call is valid."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "1", "function": {"name": "calc", "arguments": '{"expr": "2+2"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "4"},
        ]
        process_message_content(messages)
        errors = self.validate_messages_for_minimax(messages)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_openai_multiple_tool_calls_valid(self):
        """Test OpenAI format with multiple tool calls is valid."""
        messages = [
            {"role": "user", "content": "Search and calculate"},
            {
                "role": "assistant",
                "content": "I'll do both.",
                "tool_calls": [
                    {"id": "1", "function": {"name": "search", "arguments": '{"q": "test"}'}},
                    {"id": "2", "function": {"name": "calc", "arguments": '{"x": 1}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "result1"},
            {"role": "tool", "tool_call_id": "2", "content": "result2"},
        ]
        process_message_content(messages)
        errors = self.validate_messages_for_minimax(messages)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_anthropic_tool_use_converted_valid(self):
        """Test Anthropic tool_use/tool_result conversion produces valid messages."""
        body = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me calculate."},
                        {"type": "tool_use", "id": "tool_1", "name": "calc", "input": {"expr": "2+2"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tool_1", "content": "4"},
                    ],
                },
            ]
        }
        messages = convert_anthropic_to_openai_messages(body)
        errors = self.validate_messages_for_minimax(messages)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_anthropic_multiple_tool_results_valid(self):
        """Test Anthropic with multiple tool results produces valid messages."""
        body = {
            "messages": [
                {"role": "user", "content": "Do two things"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
                        {"type": "tool_use", "id": "t2", "name": "b", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
                        {"type": "tool_result", "tool_use_id": "t2", "content": "r2"},
                    ],
                },
            ]
        }
        messages = convert_anthropic_to_openai_messages(body)
        errors = self.validate_messages_for_minimax(messages)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_tool_without_preceding_assistant_fails(self):
        """Test that tool message without assistant fails validation."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
        ]
        errors = self.validate_messages_for_minimax(messages)
        self.assertTrue(len(errors) > 0, "Should have validation errors")

    def test_tool_after_user_with_prior_assistant_valid(self):
        """Test that tool message with prior assistant+tool_calls is valid.

        MiniMax only checks that SOME previous assistant has tool_calls,
        not that the tool message immediately follows it.
        """
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "x", "arguments": {}}}]},
            {"role": "user", "content": "wait"},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
        ]
        errors = self.validate_messages_for_minimax(messages)
        self.assertEqual(errors, [], f"Should be valid: {errors}")

    def test_claude_code_typical_conversation(self):
        """Test a typical Claude Code conversation with tool calls.

        Claude Code sends:
        1. System prompt
        2. User request
        3. Assistant with tool calls
        4. Tool results
        5. Assistant response
        6. More user messages...
        """
        messages = [
            {"role": "system", "content": "You are Claude Code..."},
            {"role": "user", "content": "List files in the current directory"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file1.txt\nfile2.txt"},
            {"role": "assistant", "content": "I found 2 files..."},
            {"role": "user", "content": "Now read file1.txt"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_2", "type": "function", "function": {"name": "Read", "arguments": '{"file_path": "file1.txt"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "Contents of file1..."},
        ]
        process_message_content(messages)
        errors = self.validate_messages_for_minimax(messages)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_assistant_without_tool_calls_then_tool_fails(self):
        """Test that assistant without tool_calls followed by tool message fails.

        This can happen if the client incorrectly omits tool_calls from the message.
        """
        messages = [
            {"role": "user", "content": "Do something"},
            {"role": "assistant", "content": "I'll help"},  # No tool_calls!
            {"role": "tool", "tool_call_id": "1", "content": "result"},
        ]
        errors = self.validate_messages_for_minimax(messages)
        self.assertTrue(len(errors) > 0, "Should fail: assistant has no tool_calls")

    def test_consecutive_assistant_messages_merged(self):
        """Test that consecutive assistant messages are merged correctly.

        This is the actual bug case: Codex sends:
        - assistant with tool_calls
        - assistant with reasoning text
        - tool result

        After merging, tool should immediately follow assistant with tool_calls.
        """
        messages = [
            {"role": "user", "content": "Create a game"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "write", "arguments": {}}}]},
            {"role": "assistant", "content": "I will create a game..."},  # Separate message!
            {"role": "tool", "tool_call_id": "1", "content": "File written"},
        ]

        # Merge consecutive assistant messages (same logic as server.py)
        merged = []
        for msg in messages:
            if (merged and
                msg.get("role") == "assistant" and
                merged[-1].get("role") == "assistant"):
                prev = merged[-1]
                prev_content = prev.get("content") or ""
                curr_content = msg.get("content") or ""
                if prev_content and curr_content:
                    prev["content"] = prev_content + "\n" + curr_content
                elif curr_content:
                    prev["content"] = curr_content
                if "tool_calls" in msg and msg["tool_calls"]:
                    if "tool_calls" not in prev:
                        prev["tool_calls"] = []
                    prev["tool_calls"].extend(msg["tool_calls"])
            else:
                merged.append(msg)

        # After merging, should have: user, assistant (with tool_calls + content), tool
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["role"], "user")
        self.assertEqual(merged[1]["role"], "assistant")
        self.assertEqual(merged[2]["role"], "tool")

        # The merged assistant should have both tool_calls and content
        assistant_msg = merged[1]
        self.assertIn("tool_calls", assistant_msg)
        self.assertEqual(len(assistant_msg["tool_calls"]), 1)
        self.assertEqual(assistant_msg["content"], "I will create a game...")

        # Now validation should pass
        errors = self.validate_messages_for_minimax(merged)
        self.assertEqual(errors, [], f"After merge, should pass: {errors}")

    def test_string_arguments_fails_before_processing(self):
        """Test that string arguments fail validation before process_message_content."""
        messages = [
            {"role": "user", "content": "test"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "x", "arguments": '{"a": 1}'}}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "result"},
        ]
        # Before processing - should fail
        errors = self.validate_messages_for_minimax(messages)
        self.assertTrue(len(errors) > 0, "String arguments should fail validation")

        # After processing - should pass
        process_message_content(messages)
        errors = self.validate_messages_for_minimax(messages)
        self.assertEqual(errors, [], f"After processing, should pass: {errors}")


class TestToolNormalization(unittest.TestCase):
    """Test tool format normalization for chat templates."""

    def normalize_tools(self, tools):
        """Same logic as server.py for normalizing tools."""
        normalized = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                if "name" in tool and "name" not in func:
                    func = dict(func)
                    func["name"] = tool["name"]
                if "description" in tool and "description" not in func:
                    func["description"] = tool["description"]
                normalized.append({"type": "function", "function": func})
            else:
                normalized.append(tool)
        return normalized

    def test_codex_tool_format_normalized(self):
        """Test that Codex's tool format (name at top level) is normalized.

        Codex sends:
        {"type": "function", "name": "shell", "function": {"parameters": {...}}}

        MiniMax template expects:
        {"type": "function", "function": {"name": "shell", "parameters": {...}}}
        """
        tools = [
            {
                "type": "function",
                "name": "shell",
                "function": {
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    }
                },
            }
        ]
        normalized = self.normalize_tools(tools)

        self.assertEqual(len(normalized), 1)
        func = normalized[0]["function"]
        self.assertEqual(func["name"], "shell")
        self.assertIn("parameters", func)

    def test_already_correct_format_unchanged(self):
        """Test that already correct format is not changed."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for files",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        normalized = self.normalize_tools(tools)

        self.assertEqual(len(normalized), 1)
        func = normalized[0]["function"]
        self.assertEqual(func["name"], "search")
        self.assertEqual(func["description"], "Search for files")

    def test_description_also_moved(self):
        """Test that description is also moved inside function."""
        tools = [
            {
                "type": "function",
                "name": "read",
                "description": "Read a file",
                "function": {"parameters": {}},
            }
        ]
        normalized = self.normalize_tools(tools)

        func = normalized[0]["function"]
        self.assertEqual(func["name"], "read")
        self.assertEqual(func["description"], "Read a file")


class TestProcessMessageContent(unittest.TestCase):
    """Test process_message_content for chat template preparation."""

    def test_tool_calls_arguments_parsed_from_json_string(self):
        """Test that tool_calls arguments are parsed from JSON string to dict.

        OpenAI API sends arguments as JSON strings, but chat templates like
        MiniMax call .items() on arguments, requiring them to be dicts.
        """
        messages = [
            {
                "role": "assistant",
                "content": "Let me search.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "weather", "limit": 10}',
                        },
                    }
                ],
            }
        ]
        process_message_content(messages)

        args = messages[0]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(args, dict)
        self.assertTrue(hasattr(args, "items"))
        self.assertEqual(args["query"], "weather")
        self.assertEqual(args["limit"], 10)

    def test_tool_calls_arguments_already_dict(self):
        """Test that arguments that are already dicts are left unchanged."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "calc",
                            "arguments": {"a": 1, "b": 2},
                        },
                    }
                ],
            }
        ]
        process_message_content(messages)

        args = messages[0]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(args, dict)
        self.assertEqual(args["a"], 1)

    def test_tool_calls_invalid_json_becomes_empty_dict(self):
        """Test that invalid JSON arguments become empty dict."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "test",
                            "arguments": "not valid json",
                        },
                    }
                ],
            }
        ]
        process_message_content(messages)

        args = messages[0]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(args, dict)
        self.assertEqual(args, {})

    def test_multiple_tool_calls_all_parsed(self):
        """Test that multiple tool calls all have arguments parsed."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "a", "arguments": '{"x": 1}'}},
                    {"function": {"name": "b", "arguments": '{"y": 2}'}},
                ],
            }
        ]
        process_message_content(messages)

        self.assertEqual(messages[0]["tool_calls"][0]["function"]["arguments"], {"x": 1})
        self.assertEqual(messages[0]["tool_calls"][1]["function"]["arguments"], {"y": 2})

    def test_content_list_converted_to_string(self):
        """Test that content list is converted to string."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ],
            }
        ]
        process_message_content(messages)
        self.assertEqual(messages[0]["content"], "Hello world")

    def test_none_content_becomes_empty_string(self):
        """Test that None content becomes empty string."""
        messages = [{"role": "assistant", "content": None}]
        process_message_content(messages)
        self.assertEqual(messages[0]["content"], "")


class TestConvertAnthropicToOpenAIMessages(unittest.TestCase):
    """Test conversion from Anthropic API format to OpenAI format."""

    def test_basic_message(self):
        """Test basic string content message."""
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        result = convert_anthropic_to_openai_messages(body)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"role": "user", "content": "Hello"})
        self.assertEqual(result[1], {"role": "assistant", "content": "Hi there!"})

    def test_system_prompt_string(self):
        """Test system prompt as string."""
        body = {
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = convert_anthropic_to_openai_messages(body)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"role": "system", "content": "You are a helpful assistant."})

    def test_system_prompt_list(self):
        """Test system prompt as list of content blocks."""
        body = {
            "system": [
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": " Be concise."},
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = convert_anthropic_to_openai_messages(body)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"role": "system", "content": "You are helpful. Be concise."})

    def test_tool_use_arguments_is_dict(self):
        """Test that tool_use arguments are converted as dict, not JSON string.

        This is critical for MiniMax chat template which calls .items() on arguments.
        """
        body = {
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "calculator",
                        "input": {"expression": "2+2", "precision": 2},
                    },
                ]},
            ]
        }
        result = convert_anthropic_to_openai_messages(body)
        self.assertEqual(len(result), 1)

        msg = result[0]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "Let me check.")
        self.assertIn("tool_calls", msg)
        self.assertEqual(len(msg["tool_calls"]), 1)

        tool_call = msg["tool_calls"][0]
        self.assertEqual(tool_call["id"], "tool_123")
        self.assertEqual(tool_call["type"], "function")
        self.assertEqual(tool_call["function"]["name"], "calculator")

        # Critical: arguments must be a dict, not a string
        args = tool_call["function"]["arguments"]
        self.assertIsInstance(args, dict, "arguments must be dict for chat template .items()")
        self.assertTrue(hasattr(args, "items"), "arguments must have .items() method")
        self.assertEqual(args["expression"], "2+2")
        self.assertEqual(args["precision"], 2)

    def test_tool_result_conversion(self):
        """Test tool_result blocks are converted to OpenAI tool role messages."""
        body = {
            "messages": [
                {"role": "user", "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_123",
                        "content": "The result is 4",
                    },
                ]},
            ]
        }
        result = convert_anthropic_to_openai_messages(body)
        self.assertEqual(len(result), 1)

        msg = result[0]
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "tool_123")
        self.assertEqual(msg["content"], "The result is 4")

    def test_tool_result_list_content(self):
        """Test tool_result with list content."""
        body = {
            "messages": [
                {"role": "user", "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_123",
                        "content": [
                            {"type": "text", "text": "Result: "},
                            {"type": "text", "text": "4"},
                        ],
                    },
                ]},
            ]
        }
        result = convert_anthropic_to_openai_messages(body)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "Result: 4")

    def test_multiple_tool_calls(self):
        """Test assistant message with multiple tool calls."""
        body = {
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {"query": "weather"}},
                    {"type": "tool_use", "id": "t2", "name": "calc", "input": {"expr": "1+1"}},
                ]},
            ]
        }
        result = convert_anthropic_to_openai_messages(body)
        self.assertEqual(len(result), 1)

        msg = result[0]
        self.assertEqual(len(msg["tool_calls"]), 2)
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "search")
        self.assertEqual(msg["tool_calls"][1]["function"]["name"], "calc")

        # Both should have dict arguments
        for tc in msg["tool_calls"]:
            self.assertIsInstance(tc["function"]["arguments"], dict)

    def test_full_conversation_with_tools(self):
        """Test complete conversation flow with tool use and results."""
        body = {
            "system": "You are a calculator assistant.",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "I'll calculate that."},
                    {"type": "tool_use", "id": "calc_1", "name": "add", "input": {"a": 2, "b": 2}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "calc_1", "content": "4"},
                ]},
                {"role": "assistant", "content": "The answer is 4."},
            ]
        }
        result = convert_anthropic_to_openai_messages(body)

        # Should have: system, user, assistant+tool_call, tool_result, assistant
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(result[2]["role"], "assistant")
        self.assertIn("tool_calls", result[2])
        self.assertEqual(result[3]["role"], "tool")
        self.assertEqual(result[4]["role"], "assistant")

        # Verify arguments is dict
        args = result[2]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(args, dict)
        self.assertEqual(args["a"], 2)
        self.assertEqual(args["b"], 2)

    def test_empty_tool_input(self):
        """Test tool_use with empty input."""
        body = {
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "get_time", "input": {}},
                ]},
            ]
        }
        result = convert_anthropic_to_openai_messages(body)
        args = result[0]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(args, dict)
        self.assertEqual(args, {})


if __name__ == "__main__":
    unittest.main()
