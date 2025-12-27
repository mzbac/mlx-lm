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


if __name__ == "__main__":
    unittest.main()
