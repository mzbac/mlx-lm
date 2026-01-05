# Copyright © 2025 Apple Inc.

"""
Distributed HTTP server for MLX LM.

Run with mlx.launch for distributed inference across multiple machines:

```bash
# Local multi-process test (2 processes on same machine)
mlx.launch -n 2 \
    --env MLX_METAL_FAST_SYNCH=1 \
    python mlx_lm/examples/distributed_server.py \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --port 8080

# Two machines with ring backend
mlx.launch --backend ring --hostfile hosts.json \
    --env MLX_METAL_FAST_SYNCH=1 \
    python mlx_lm/examples/distributed_server.py \
    --model mlx-community/Llama-3.3-70B-Instruct-4bit \
    --host 0.0.0.0 --port 8080
```

Architecture:
- All ranks run the generation loop simultaneously (required for pipeline parallelism)
- Rank 0 receives HTTP requests and broadcasts prompts to all ranks
- All ranks call model forward together for each token
- Only rank 0 returns HTTP responses

Supports all server.py functionality:
- OpenAI-compatible /v1/chat/completions and /v1/completions
- Anthropic-compatible /v1/messages
- Tool calling (MiniMax XML, Devstral formats)
- Streaming responses

For more information on running distributed programs with MLX see:
https://ml-explore.github.io/mlx/build/html/usage/distributed.html
"""

import argparse
import copy
import json
import logging
import os
import re
import socket
import time
import uuid
import warnings
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue, Empty
from socketserver import ThreadingMixIn
from threading import Thread, Lock
from typing import Optional, List, Tuple, Any

import mlx.core as mx

from mlx_lm import stream_generate
from mlx_lm.utils import sharded_load
from mlx_lm.models.cache import (
    KVCache,
    make_prompt_cache,
    can_trim_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.sample_utils import make_sampler, make_logits_processors


# Regexes used in hot paths (tool-call parsing/extraction).
MINIMAX_INVOKE_RE = re.compile(
    r'<invoke\s+name=["\']([^"\']+)["\']>(.*?)</invoke>',
    re.DOTALL,
)
MINIMAX_PARAM_RE = re.compile(
    r'<parameter\s+name=["\']([^"\']+)["\']>(.*?)</parameter>',
    re.DOTALL,
)
MINIMAX_TOOL_CALL_WRAPPER_RE = re.compile(
    r"<minimax:tool_call>(.*?)(?:</minimax:tool_call>|\[e~\[|$)",
    re.DOTALL,
)
MINIMAX_INVOKE_BLOCK_RE = re.compile(
    r'<invoke\s+name=["\'][^"\']+["\']>.*?</invoke>',
    re.DOTALL,
)
DEVSTRAL_TOOL_CALL_RE = re.compile(
    r"\[TOOL_CALLS\](\w+)\[ARGS\](\{.*?\})(?=\[TOOL_CALLS\]|</s>|$)",
    re.DOTALL,
)
DEVSTRAL_TOOL_CALL_BLOCK_RE = re.compile(
    r"\[TOOL_CALLS\]\w+\[ARGS\]\{.*?\}(?=\[TOOL_CALLS\]|</s>|$)",
    re.DOTALL,
)

# Default decoding controls (MiniMax-focused discovery mode).
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_REPETITION_CONTEXT_SIZE = 256


def _minimax_is_path_like(value: str) -> bool:
    # MiniMax frequently inserts spaces around path separators in tool-call args.
    # Only normalize strings that look like file paths (absolute, home, or dot-relative).
    return value.startswith(("/", "~/", "./", "../"))


def _minimax_normalize_path_separators(value: str) -> str:
    # Remove tokenizer-inserted spaces around "/" and "." (e.g. "/ Users / foo .py").
    # Keep this conservative and only touch path-like strings.
    if not value or not _minimax_is_path_like(value):
        return value
    value = re.sub(r"\s*/\s*", "/", value)
    value = re.sub(r"\s*\.\s*", ".", value)
    return value


def _minimax_normalize_hyphen_spacing(value: str) -> str:
    # MiniMax can emit tokenizer-leading spaces after "-" inside identifiers (e.g. "high- score-label").
    # Avoid touching " - " minus operators / list bullets by only fixing cases where "-" is directly
    # attached to the left token.
    if not value:
        return value
    # "button-- primary" -> "button--primary"
    value = re.sub(r"(?<=\w-)-\s+(?=\w)", "-", value)
    # "high- score" -> "high-score"
    value = re.sub(r"(?<=\w)-\s+(?=\w)", "-", value)
    return value


def _minimax_normalize_underscore_spacing(value: str) -> str:
    # MiniMax can emit tokenizer-leading spaces after "_" inside identifiers/paths
    # (e.g. "oauth_ server.rs"). Avoid touching " _ " by only fixing cases where
    # "_" is directly attached to the left token.
    if not value:
        return value
    value = re.sub(r"(?<=\w)_\s+(?=\w)", "_", value)
    return value


def _minimax_normalize_tokenizer_spacing(value: str) -> str:
    value = _minimax_normalize_path_separators(value)
    value = _minimax_normalize_hyphen_spacing(value)
    value = _minimax_normalize_underscore_spacing(value)
    return value


def _minimax_normalize_tool_arguments(obj: Any) -> Any:
    if isinstance(obj, str):
        return _minimax_normalize_tokenizer_spacing(obj)
    if isinstance(obj, list):
        return [_minimax_normalize_tool_arguments(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _minimax_normalize_tool_arguments(v) for k, v in obj.items()}
    return obj


def parse_minimax_tool_calls(tool_text: str) -> List[dict]:
    """Parse MiniMax-style XML tool calls into standard JSON format."""
    tool_calls = []

    for match in MINIMAX_INVOKE_RE.finditer(tool_text):
        tool_name = match.group(1)
        invoke_body = match.group(2)
        arguments = {}
        for param_match in MINIMAX_PARAM_RE.finditer(invoke_body):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()
            try:
                parsed = json.loads(param_value)
                arguments[param_name] = _minimax_normalize_tool_arguments(parsed)
            except (json.JSONDecodeError, ValueError):
                arguments[param_name] = _minimax_normalize_tokenizer_spacing(param_value)
        tool_calls.append({"name": tool_name, "arguments": arguments})

    return tool_calls


def parse_devstral_tool_calls(text: str) -> List[dict]:
    """Parse Devstral-style [TOOL_CALLS]name[ARGS]json tool calls."""
    tool_calls = []

    for match in DEVSTRAL_TOOL_CALL_RE.finditer(text):
        func_name = match.group(1)
        args_str = match.group(2)
        try:
            arguments = json.loads(args_str)
        except (json.JSONDecodeError, ValueError):
            brace_count = 0
            end_idx = 0
            for i, char in enumerate(args_str):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            if end_idx > 0:
                try:
                    arguments = json.loads(args_str[:end_idx])
                except (json.JSONDecodeError, ValueError):
                    arguments = {}
            else:
                arguments = {}
        tool_calls.append({"name": func_name, "arguments": arguments})

    return tool_calls


def normalize_tool_calls(tool_calls: List[Any]) -> List[dict]:
    """Normalize tool call entries to a list of {name, arguments} dicts."""
    normalized: List[dict] = []
    for entry in tool_calls or []:
        if isinstance(entry, dict):
            if "name" in entry and "arguments" in entry:
                args = entry.get("arguments")
                normalized.append({**entry, "arguments": _minimax_normalize_tool_arguments(args)})
            continue
        if entry is None:
            continue

        text = str(entry).strip()
        if not text:
            continue

        if "<invoke" in text and "</invoke>" in text:
            normalized.extend(parse_minimax_tool_calls(text))
            continue

        if "[TOOL_CALLS]" in text:
            normalized.extend(parse_devstral_tool_calls(text))
            continue

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue

        if isinstance(parsed, dict):
            if "arguments" in parsed:
                parsed["arguments"] = _minimax_normalize_tool_arguments(parsed.get("arguments"))
            normalized.append(parsed)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    if "arguments" in item:
                        item = {**item, "arguments": _minimax_normalize_tool_arguments(item.get("arguments"))}
                    normalized.append(item)

    return normalized


def filter_by_schema(value: Any, schema: dict) -> Any:
    """Recursively filter a value to only include properties allowed by schema."""
    if schema is None:
        return value

    # Handle schema unions like {"type": ["string", "null"]}.
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        # If null is allowed and value is None, preserve None.
        if value is None and "null" in schema_type:
            return None
        # Prefer the first non-null type as the coercion target.
        schema_type = next((t for t in schema_type if t != "null"), schema_type[0] if schema_type else None)

    # Coerce to string when required (opencode tools often require strict strings, e.g. write.content).
    if schema_type == "string":
        if isinstance(value, str):
            return value
        try:
            coerced = json.dumps(value, ensure_ascii=False)
        except Exception:
            coerced = str(value)
        return coerced

    if schema_type == "object":
        if not isinstance(value, dict):
            return value
        properties = schema.get("properties", {})
        additional_props = schema.get("additionalProperties", True)

        if additional_props is False:
            filtered = {}
            for key, val in value.items():
                if key in properties:
                    filtered[key] = filter_by_schema(val, properties[key])
            return filtered

        filtered = {}
        for key, val in value.items():
            if key in properties:
                filtered[key] = filter_by_schema(val, properties[key])
            else:
                filtered[key] = val
        return filtered

    if schema_type == "array":
        if not isinstance(value, list):
            return value
        items_schema = schema.get("items", {})
        return [filter_by_schema(item, items_schema) for item in value]

    return value


def filter_tool_call_by_schema(tool_call: dict, tools: Optional[List[dict]]) -> dict:
    """Filter tool call arguments to match the tool's schema."""
    if not tools:
        return tool_call

    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})

    def schema_expects_string(s: Any) -> bool:
        if not isinstance(s, dict):
            return False
        t = s.get("type")
        if t == "string":
            return True
        if isinstance(t, list) and "string" in t:
            return True
        any_of = s.get("anyOf")
        if isinstance(any_of, list):
            return any(schema_expects_string(o) for o in any_of)
        one_of = s.get("oneOf")
        if isinstance(one_of, list):
            return any(schema_expects_string(o) for o in one_of)
        return False

    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            name = func.get("name")
            schema = func.get("parameters", {})
        else:
            name = tool.get("name")
            schema = tool.get("parameters", {})

        if name == tool_name and schema:
            filtered_args = filter_by_schema(arguments, schema)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                try:
                    props = schema.get("properties", {}) if schema.get("type") == "object" else {}
                    if isinstance(arguments, dict) and isinstance(filtered_args, dict) and isinstance(props, dict):
                        for k, prop_schema in props.items():
                            if not schema_expects_string(prop_schema):
                                continue
                            before = arguments.get(k)
                            after = filtered_args.get(k)
                            if before is not None and not isinstance(before, str) and isinstance(after, str):
                                logging.debug(
                                    f"Coerced tool arg to string: tool={tool_name}, arg={k}, from={type(before).__name__}"
                                )
                except Exception:
                    logging.debug("Failed to debug tool arg coercions", exc_info=True)
            return {"name": tool_name, "arguments": filtered_args}

    return tool_call


def process_message_content(messages):
    """
    Convert message content to a format suitable for `apply_chat_template`.

    Converts content lists to strings and parses tool_call arguments from JSON
    strings to dicts (needed for templates that call .items() on arguments).
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text_fragments = [
                fragment["text"] for fragment in content if fragment["type"] == "text"
            ]
            if len(text_fragments) != len(content):
                raise ValueError("Only 'text' content type is supported.")
            message["content"] = "".join(text_fragments)
        elif content is None:
            message["content"] = ""

        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                func = tool_call.get("function", tool_call)
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        func["arguments"] = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        func["arguments"] = {}


def convert_anthropic_to_openai_messages(body: dict) -> List[dict]:
    """Convert Anthropic API messages to OpenAI format for chat templates."""
    messages = []

    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = "".join(b.get("text", "") for b in system if b.get("type") == "text")
            if text:
                messages.append({"role": "system", "content": text})

    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            tool_calls = []
            tool_results = []

            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": block.get("input", {}),
                        },
                    })
                elif block_type == "tool_result":
                    tool_id = block.get("tool_use_id", "")
                    result = block.get("content", "")
                    if isinstance(result, list):
                        result = "".join(
                            b.get("text", "") for b in result if b.get("type") == "text"
                        )
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result,
                    })

            if role == "assistant":
                msg_dict = {"role": "assistant", "content": "".join(text_parts) or None}
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                messages.append(msg_dict)
            elif tool_results:
                for tool_result in tool_results:
                    messages.append(tool_result)
            else:
                messages.append({"role": role, "content": "".join(text_parts)})
        else:
            messages.append({"role": role, "content": ""})

    return messages


def convert_anthropic_tools(tools: Optional[List[dict]]) -> Optional[List[dict]]:
    """Convert Anthropic tool format to OpenAI format."""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }
        for tool in tools
    ]


class LRUPromptCache:
    """LRU cache for prompt prefixes to speed up generation."""

    @dataclass
    class CacheEntry:
        prompt_cache: List[Any]
        count: int

    @dataclass
    class SearchResult:
        model: Any
        exact: List[int]
        shorter: List[int]
        longer: List[int]
        common_prefix: int

    def __init__(self, max_size: int = 2):
        self.max_size = max_size
        self._cache = {}
        self._lru = deque()

    def _search(self, model, tokens):
        """Search the cache for a prompt cache. Return exact or close match."""
        if model not in self._cache:
            return self.SearchResult(model, None, None, None, 0)

        current = self._cache[model]
        last_cache_index = -1
        index = 0

        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "cache" in current:
                last_cache_index = index
            index += 1

        if last_cache_index == len(tokens) - 1:
            return self.SearchResult(model, tokens, None, None, 0)

        shorter = None
        if last_cache_index > 0:
            shorter = tokens[: last_cache_index + 1]

        longer = None
        common_prefix = index
        if index > 0 and last_cache_index <= 0:
            best = None
            stack = [(current, [])]
            while stack:
                current, extra = stack.pop()
                if "cache" in current:
                    if best is None or len(extra) < len(best):
                        best = extra
                else:
                    for tok in current:
                        stack.append((current[tok], extra + [tok]))
            longer = tokens[:index] + best
        return self.SearchResult(model, None, shorter, longer, common_prefix)

    def _get(self, model, tokens):
        current = self._cache[model]
        for tok in tokens:
            current = current[tok]
        return current["cache"]

    def _delete(self, model, tokens):
        path = [self._cache[model]]
        for tok in tokens:
            path.append(path[-1][tok])
        del path[-1]["cache"]
        for i in reversed(range(len(tokens))):
            d_prev, d, t = path[i], path[i + 1], tokens[i]
            if len(d) > 0:
                break
            del d_prev[t]

    def _extract(self, model, tokens):
        cache_entry = self._get(model, tokens)
        if cache_entry.count == 1:
            self._delete(model, tokens)
            self._lru.remove((model, tokens))
            return cache_entry

        cache_entry.count -= 1
        return self.CacheEntry(
            copy.deepcopy(cache_entry.prompt_cache),
            1,
        )

    def fetch_nearest_cache(self, model, tokens):
        result = self._search(model, tokens)
        if result.exact is not None:
            cache_entry = self._extract(result.model, result.exact)
            return cache_entry.prompt_cache, []

        if result.shorter is not None:
            cache_entry = self._extract(result.model, result.shorter)
            prefix_len = len(result.shorter)
            return cache_entry.prompt_cache, tokens[prefix_len:]

        if result.longer is not None:
            cache_entry = self._get(result.model, result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                cache_entry = self.CacheEntry(
                    copy.deepcopy(cache_entry.prompt_cache),
                    1,
                )
                prefix = min(len(tokens) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                trim_prompt_cache(cache_entry.prompt_cache, num_to_trim)
                return cache_entry.prompt_cache, tokens[prefix:]

        return None, tokens

    def insert_cache(self, model, tokens, prompt_cache):
        if model not in self._cache:
            self._cache[model] = {}
        current = self._cache[model]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]

        if "cache" in current:
            current["cache"].count += 1
            self._lru.remove((model, tokens))
        else:
            current["cache"] = self.CacheEntry(prompt_cache, 1)

        self._lru.append((model, tokens))
        if len(self._lru) > self.max_size:
            model, tokens = self._lru.popleft()
            self._delete(model, tokens)


# Maximum prompt length for broadcasting (in tokens)
MAX_PROMPT_LENGTH = 131072

# Stop sequence broadcast limits
MAX_STOP_SEQUENCES = 8
MAX_STOP_SEQUENCE_LENGTH = 256


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Thread-per-request HTTP server."""
    daemon_threads = True


class DistributedState:
    """Coordinates generation requests across distributed ranks.

    For pipeline parallelism to work, all ranks must call the model forward pass
    together. This class provides synchronization primitives to broadcast prompts
    from rank 0 to all ranks.
    """

    def __init__(self, group):
        self.group = group
        self.rank = group.rank()
        self.world_size = group.size()
        self.request_queue = Queue()  # Only used by rank 0
        self.lock = Lock()

    def broadcast_request(self):
        """Broadcast request from rank 0 to all ranks.

        Returns (prompt_tokens, max_tokens, seed, temperature, top_p, top_k, stop_token_sequences, response_queue, request)
        or (None, 0, 0, 0.0, 0.0, 0, [], None, None) if no request.

        Uses all_sum for broadcasting since it's available on all backends.
        """
        prompt_tokens = None
        max_tokens = 256
        seed = 0
        temperature = 0.0
        top_p = 0.0
        top_k = 0
        repetition_penalty = DEFAULT_REPETITION_PENALTY
        repetition_context_size = DEFAULT_REPETITION_CONTEXT_SIZE
        stop_token_sequences = []
        response_queue = None
        request = None

        # Rank 0 checks for new request
        if self.rank == 0:
            try:
                request = self.request_queue.get_nowait()
                prompt_tokens = request["prompt_tokens"]
                max_tokens = request["max_tokens"]
                seed = int(request.get("seed") or 0)
                temperature = float(request.get("temperature") or 0.0)
                top_p = float(request.get("top_p") or 0.0)
                top_k = int(request.get("top_k") or 0)
                rp = request.get("repetition_penalty", repetition_penalty)
                repetition_penalty = float(repetition_penalty if rp is None else rp)
                rcs = request.get("repetition_context_size", repetition_context_size)
                repetition_context_size = int(
                    repetition_context_size if rcs is None else rcs
                )
                stop_token_sequences = request.get("stop_token_sequences") or []
                response_queue = request["response_queue"]
                logging.info(
                    "Broadcasting request: prompt_len=%d, max_tokens=%d, stop_sequences=%d",
                    len(prompt_tokens) if prompt_tokens else 0,
                    int(max_tokens) if max_tokens is not None else 0,
                    len(stop_token_sequences or []),
                )
            except Empty:
                prompt_tokens = None

        # Broadcast metadata first so idle polling only does one collective.
        # Metadata: [length, max_tokens, seed, top_k, stop_count, repetition_context_size]
        if self.rank == 0:
            length = len(prompt_tokens) if prompt_tokens else 0
            if length > MAX_PROMPT_LENGTH:
                length = MAX_PROMPT_LENGTH
            stop_count = min(len(stop_token_sequences or []), MAX_STOP_SEQUENCES)
            meta = mx.array(
                [length, max_tokens, seed, top_k, stop_count, repetition_context_size],
                dtype=mx.int32,
            )
        else:
            meta = mx.zeros((6,), dtype=mx.int32)

        t0 = time.perf_counter()
        meta = mx.distributed.all_sum(meta, stream=mx.cpu)
        mx.eval(meta)
        meta_dt = time.perf_counter() - t0

        length = int(meta[0].item())
        max_tokens = int(meta[1].item())
        seed = int(meta[2].item())
        top_k = int(meta[3].item())
        stop_count = int(meta[4].item())
        repetition_context_size = int(meta[5].item())

        # Broadcast floats: [temperature, top_p]
        if length == 0:
            return (
                None,
                0,
                0,
                0.0,
                0.0,
                0,
                0.0,
                DEFAULT_REPETITION_CONTEXT_SIZE,
                [],
                None,
                None,
            )

        if self.rank == 0:
            meta_f = mx.array([temperature, top_p, repetition_penalty], dtype=mx.float32)
        else:
            meta_f = mx.zeros((3,), dtype=mx.float32)

        t1 = time.perf_counter()
        meta_f = mx.distributed.all_sum(meta_f, stream=mx.cpu)
        mx.eval(meta_f)
        meta_f_dt = time.perf_counter() - t1
        temperature = float(meta_f[0].item())
        top_p = float(meta_f[1].item())
        repetition_penalty = float(meta_f[2].item())

        # Broadcast actual prompt tokens
        if self.rank == 0:
            tokens = mx.array(prompt_tokens[:length], dtype=mx.int32)
        else:
            tokens = mx.zeros((length,), dtype=mx.int32)

        t2 = time.perf_counter()
        tokens = mx.distributed.all_sum(tokens, stream=mx.cpu)
        mx.eval(tokens)
        tokens_dt = time.perf_counter() - t2

        prompt = tokens.tolist()

        stop_token_sequences_out = []
        if stop_count > 0:
            # Broadcast stop token sequences (padded per-sequence)
            if self.rank == 0:
                lens = [0] * stop_count
                flat = [0] * (stop_count * MAX_STOP_SEQUENCE_LENGTH)
                for i, seq in enumerate((stop_token_sequences or [])[:stop_count]):
                    if not seq:
                        continue
                    seq = [int(x) for x in seq]
                    l = min(len(seq), MAX_STOP_SEQUENCE_LENGTH)
                    lens[i] = l
                    start = i * MAX_STOP_SEQUENCE_LENGTH
                    flat[start : start + l] = seq[:l]
                stop_lens = mx.array(lens, dtype=mx.int32)
                stop_tokens = mx.array(flat, dtype=mx.int32)
            else:
                stop_lens = mx.zeros((stop_count,), dtype=mx.int32)
                stop_tokens = mx.zeros((stop_count * MAX_STOP_SEQUENCE_LENGTH,), dtype=mx.int32)

            t3 = time.perf_counter()
            stop_lens = mx.distributed.all_sum(stop_lens, stream=mx.cpu)
            stop_tokens = mx.distributed.all_sum(stop_tokens, stream=mx.cpu)
            mx.eval(stop_lens, stop_tokens)
            stop_dt = time.perf_counter() - t3

            for i in range(stop_count):
                l = int(stop_lens[i].item())
                if l <= 0:
                    continue
                start = i * MAX_STOP_SEQUENCE_LENGTH
                stop_token_sequences_out.append(stop_tokens[start : start + l].tolist())

        if self.rank == 0:
            logging.info(
                "Broadcast timings: meta=%.3fs floats=%.3fs tokens=%.3fs stop=%.3fs (length=%d stop_count=%d)",
                meta_dt,
                meta_f_dt,
                tokens_dt,
                stop_dt if stop_count > 0 else 0.0,
                length,
                stop_count,
            )

        return (
            prompt,
            max_tokens,
            seed,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            repetition_context_size,
            stop_token_sequences_out,
            response_queue,
            request,
        )


def build_kmp_lps(pattern: List[int]) -> List[int]:
    """Build KMP LPS table for token stop-sequence matching."""
    lps = [0] * len(pattern)
    length = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length != 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    return lps


def extract_tool_calls(text: str, tools: Optional[List] = None) -> Tuple[str, List[dict]]:
    """Extract tool calls from generated text and return cleaned text."""
    extracted: List[dict] = []
    clean_text = text

    if not text:
        return "", []

    def extract_and_filter(tool_text: str) -> List[dict]:
        result: List[dict] = []
        for p in parse_minimax_tool_calls(tool_text):
            if tools:
                p = filter_tool_call_by_schema(p, tools)
            result.append(p)
        return result

    def extract_and_filter_devstral(full_text: str) -> List[dict]:
        result: List[dict] = []
        for p in parse_devstral_tool_calls(full_text):
            if tools:
                p = filter_tool_call_by_schema(p, tools)
            result.append(p)
        return result

    if text and "[TOOL_CALLS]" in text:
        extracted.extend(extract_and_filter_devstral(text))
        if extracted:
            clean_text = DEVSTRAL_TOOL_CALL_BLOCK_RE.sub("", text)

    elif text and "<minimax:tool_call>" in text:
        for match in MINIMAX_TOOL_CALL_WRAPPER_RE.finditer(text):
            extracted.extend(extract_and_filter(match.group(1).strip()))
        clean_text = MINIMAX_TOOL_CALL_WRAPPER_RE.sub("", text)

        if not extracted and "<invoke" in text:
            extracted.extend(extract_and_filter(text))
            if extracted:
                clean_text = MINIMAX_INVOKE_BLOCK_RE.sub("", clean_text)

    elif text and "<invoke" in text and "</invoke>" in text:
        extracted.extend(extract_and_filter(text))
        if extracted:
            clean_text = MINIMAX_INVOKE_BLOCK_RE.sub("", text)

    if clean_text:
        clean_text = re.sub(r"</?minimax:tool_call>", "", clean_text)
        clean_text = re.sub(
            r"\[TOOL_CALLS\]|\[ARGS\]|\[/TOOL_RESULTS\]|\[TOOL_RESULTS\]", "", clean_text
        )

    if extracted:
        try:
            names = [t.get("name") for t in extracted if isinstance(t, dict)]
        except Exception:
            names = []
        logging.info(f"Extracted {len(extracted)} tool calls: {names}")

    return clean_text, extracted







def build_tool_calls_payload(tool_calls: List[Any]) -> Optional[List[dict]]:
    """Convert tool call text entries into OpenAI tool_calls payload."""
    normalized = normalize_tool_calls(tool_calls or [])
    if not normalized:
        return None

    payload = []
    for i, tool_call in enumerate(normalized):
        args = tool_call.get("arguments", "")
        if isinstance(args, str):
            arguments = args
        else:
            arguments = json.dumps(args)
        payload.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": tool_call.get("name", None),
                "arguments": arguments,
            },
            "index": i,
        })
    return payload or None


class DistributedHandler(BaseHTTPRequestHandler):
    """HTTP request handler for distributed inference."""

    def __init__(self, dist_state, tokenizer, args, *handler_args, **handler_kwargs):
        self.dist_state = dist_state
        self.tokenizer = tokenizer
        self.args = args
        super().__init__(*handler_args, **handler_kwargs)

    def log_message(self, format, *args):
        logging.debug(f"[HTTP] {format % args}")

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {"status": "ok"})
        elif self.path.startswith("/v1/models"):
            self._json_response(200, {
                "object": "list",
                "data": [{"id": self.args.model, "object": "model", "owned_by": "mlx"}]
            })
        else:
            self.send_error(404)

    def do_POST(self):
        path = self.path.split("?")[0]  # Remove query string

        handlers = {
            "/v1/chat/completions": self._handle_chat,
            "/chat/completions": self._handle_chat,
            "/v1/completions": self._handle_text,
            "/v1/messages": self._handle_anthropic,
        }

        handler = handlers.get(path)
        if handler:
            try:
                handler()
            except Exception as e:
                logging.exception("Request error")
                self._json_response(500, {"error": str(e)})
        else:
            self.send_error(404)

    def _parse_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length).decode())

    def _json_response(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _stream_response(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._set_cors_headers()
        self.end_headers()

    def _handle_chat(self):
        body = self._parse_body()
        messages = body.get("messages", [])
        tools = body.get("tools")
        stream = body.get("stream", False)
        stream_options = body.get("stream_options", None)
        max_tokens = body.get("max_tokens", self.args.max_tokens)
        temperature = body.get("temperature", self.args.temperature)
        top_p = body.get("top_p", body.get("topP", self.args.top_p))
        top_k = body.get("top_k", body.get("topK", self.args.top_k))
        repetition_penalty = body.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)
        repetition_context_size = body.get(
            "repetition_context_size", DEFAULT_REPETITION_CONTEXT_SIZE
        )
        seed = body.get("seed", None)
        stop_words = body.get("stop") or []
        model = body.get("model", self.args.model)
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            temperature = float(self.args.temperature)
        try:
            top_p = float(top_p)
        except (TypeError, ValueError):
            top_p = float(self.args.top_p)
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = int(self.args.top_k)
        try:
            repetition_penalty = float(repetition_penalty)
        except (TypeError, ValueError):
            repetition_penalty = DEFAULT_REPETITION_PENALTY
        try:
            repetition_context_size = int(repetition_context_size)
        except (TypeError, ValueError):
            repetition_context_size = DEFAULT_REPETITION_CONTEXT_SIZE
        if temperature < 0:
            temperature = float(self.args.temperature)
        if top_p < 0 or top_p > 1:
            top_p = float(self.args.top_p)
        if top_k < 0:
            top_k = int(self.args.top_k)
        if repetition_penalty < 0:
            repetition_penalty = DEFAULT_REPETITION_PENALTY
        if repetition_context_size < 0:
            repetition_context_size = DEFAULT_REPETITION_CONTEXT_SIZE
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                seed = None
        if seed is None:
            seed = int(time.time_ns() & 0x7FFFFFFF)
        if isinstance(stop_words, str):
            stop_words = [stop_words]
        if not isinstance(stop_words, list):
            stop_words = []
        stop_words = [s for s in stop_words if isinstance(s, str) and s]
        stop_token_sequences = []
        for sw in stop_words[:MAX_STOP_SEQUENCES]:
            try:
                seq = self.tokenizer.encode(sw, add_special_tokens=False)
            except TypeError:
                seq = self.tokenizer.encode(sw)
            if seq:
                stop_token_sequences.append(seq)

        logging.info(f"Received request: model={model}, max_tokens={max_tokens}, stream={stream}, num_messages={len(messages)}")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            try:
                tool_names = []
                if isinstance(tools, list):
                    for t in tools:
                        if not isinstance(t, dict):
                            continue
                        func = t.get("function")
                        if isinstance(func, dict) and isinstance(func.get("name"), str):
                            tool_names.append(func["name"])
                        elif isinstance(t.get("name"), str):
                            tool_names.append(t["name"])

                msg_preview = []
                for m in (messages or [])[:5]:
                    if not isinstance(m, dict):
                        continue
                    content = m.get("content")
                    if isinstance(content, str):
                        content_preview = content[:200]
                    else:
                        content_preview = str(content)[:200]
                    msg_preview.append({"role": m.get("role"), "content": content_preview})

                debug_obj = {
                    "model": model,
                    "stream": stream,
                    "max_tokens": max_tokens,
                    "stream_options": stream_options,
                    "tools": tool_names,
                    "messages_preview": msg_preview,
                }
                logging.debug(f"Request body preview: {json.dumps(debug_obj, ensure_ascii=False)}")
            except Exception:
                logging.debug("Failed to render request body preview", exc_info=True)

        # Map Claude model names to local
        if model.startswith(("claude-", "anthropic")):
            model = self.args.model

        process_message_content(messages)
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False
        )
        emit_initial_think = prompt.rstrip().endswith("<think>")
        prompt_tokens = self.tokenizer.encode(prompt)
        
        logging.info(f"Processing prompt: {len(prompt_tokens)} tokens")

        response_queue = Queue()
        self.dist_state.request_queue.put({
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "repetition_context_size": repetition_context_size,
            "stop_token_sequences": stop_token_sequences,
            "response_queue": response_queue,
            "tools": tools,
        })

        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        if stream:
            self._stream_chat(response_queue, request_id, model, tools, stream_options, emit_initial_think)
        else:
            self._blocking_chat(response_queue, request_id, model, tools, emit_initial_think)

    def _stream_chat(self, queue, request_id, model, tools, stream_options, emit_initial_think: bool = False):
        self._stream_response()
        # Emit an initial SSE comment so clients see bytes immediately (helps avoid idle timeouts
        # during long prefill before the first generated token).
        try:
            self.wfile.write(b": keepalive\n\n")
            self.wfile.flush()
        except Exception:
            pass

        has_tool_calling = getattr(self.tokenizer, "has_tool_calling", False)
        tool_call_start = getattr(self.tokenizer, "tool_call_start", None)
        tool_call_end = getattr(self.tokenizer, "tool_call_end", None)

        in_tool_call = False
        tool_calls = []
        tool_text = ""

        in_xml_tool_call = False
        xml_tool_buffer = ""
        content_buffer = ""
        finish_reason = None
        prompt_toks = 0
        gen_toks = 0

        xml_start_markers = ["<minimax:tool_call>", "[TOOL_CALLS]", "<invoke"]
        holdback_len = max(len(m) for m in xml_start_markers) - 1

        def find_first_marker(text: str) -> Optional[Tuple[int, str]]:
            best = None
            for marker in xml_start_markers:
                idx = text.find(marker)
                if idx != -1 and (best is None or idx < best[0]):
                    best = (idx, marker)
            return best

        def send_chunk(content: str, tool_call_texts: Optional[List[str]] = None):
            nonlocal tool_calls
            content_to_send = content if content else ""
            chunk_tool_calls = build_tool_calls_payload(tool_call_texts) if tool_call_texts else None

            if not content_to_send and not chunk_tool_calls:
                return

            if chunk_tool_calls:
                for tc in chunk_tool_calls:
                    logging.info(
                        f"Streaming tool call {tc.get('index')}: {tc.get('function', {}).get('name')}"
                    )

            delta = {"role": "assistant", "content": content_to_send}
            if chunk_tool_calls:
                delta["tool_calls"] = chunk_tool_calls

            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            tool_calls = []

        try:
            if emit_initial_think:
                send_chunk("<think>\n")
            while True:
                try:
                    item = queue.get(timeout=10)
                except Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue

                if item is None:
                    break

                gen_text = item.get("text", "")
                finish_reason = item.get("finish_reason")
                prompt_toks = item.get("prompt_tokens", prompt_toks)
                gen_toks = item.get("generation_tokens", gen_toks)

                if has_tool_calling and gen_text == tool_call_start:
                    in_tool_call = True
                elif in_tool_call:
                    if gen_text == tool_call_end:
                        tool_calls.append(tool_text)
                        tool_text = ""
                        in_tool_call = False
                    else:
                        tool_text += gen_text
                else:
                    if in_xml_tool_call:
                        xml_tool_buffer += gen_text
                    else:
                        content_buffer += gen_text

                        marker = find_first_marker(content_buffer)
                        if marker is not None:
                            idx, which = marker
                            logging.debug(
                                f"Detected tool-call marker {which!r} at idx={idx}, buffer_len={len(content_buffer)}"
                            )
                            safe_prefix = content_buffer[:idx]
                            if safe_prefix or tool_calls:
                                send_chunk(safe_prefix, tool_calls if tool_calls else None)
                            in_xml_tool_call = True
                            xml_tool_buffer = content_buffer[idx:]
                            content_buffer = ""
                        else:
                            if holdback_len > 0 and len(content_buffer) > holdback_len:
                                flushable = content_buffer[:-holdback_len]
                                content_buffer = content_buffer[-holdback_len:]
                                if flushable or tool_calls:
                                    send_chunk(flushable, tool_calls if tool_calls else None)

            clean_from_xml, extracted = ("", [])
            if in_xml_tool_call and xml_tool_buffer:
                logging.debug(f"Parsing buffered tool-call text (len={len(xml_tool_buffer)}).")
                clean_from_xml, extracted = extract_tool_calls(xml_tool_buffer, tools)

            clean_text = content_buffer + (clean_from_xml or "")
            clean_text = clean_text if clean_text else ""
            all_tool_calls = tool_calls + extracted

            final_tool_calls = build_tool_calls_payload(all_tool_calls) if all_tool_calls else None
            if final_tool_calls:
                for tc in final_tool_calls:
                    logging.info(
                        f"Streaming tool call {tc.get('index')}: {tc.get('function', {}).get('name')}"
                    )

            final_finish = finish_reason or "stop"
            if final_tool_calls and final_finish == "stop":
                final_finish = "tool_calls"

            delta = {"role": "assistant", "content": clean_text}
            if final_tool_calls:
                delta["tool_calls"] = final_tool_calls

            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": final_finish}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

            if stream_options and stream_options.get("include_usage"):
                usage_chunk = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_toks,
                        "completion_tokens": gen_toks,
                        "total_tokens": prompt_toks + gen_toks,
                    },
                }
                self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
                self.wfile.flush()

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _blocking_chat(self, queue, request_id, model, tools, emit_initial_think: bool = False):
        full_text = ""
        tool_calls = []
        tool_text = ""
        in_tool_call = False
        has_tool_calling = getattr(self.tokenizer, "has_tool_calling", False)
        tool_call_start = getattr(self.tokenizer, "tool_call_start", None)
        tool_call_end = getattr(self.tokenizer, "tool_call_end", None)
        finish_reason = None
        prompt_toks = 0
        gen_toks = 0

        blocking_timeout_s = float(os.environ.get("DISTRIBUTED_BLOCKING_TIMEOUT_S", "3600"))
        blocking_poll_s = float(os.environ.get("DISTRIBUTED_BLOCKING_POLL_S", "1"))
        start_t = time.perf_counter()

        while True:
            if blocking_timeout_s > 0:
                try:
                    item = queue.get(timeout=blocking_poll_s)
                except Empty:
                    if (time.perf_counter() - start_t) >= blocking_timeout_s:
                        logging.error(
                            "Blocking chat request timed out after %.1fs (id=%s model=%s)",
                            blocking_timeout_s,
                            request_id,
                            model,
                        )
                        self._json_response(
                            504,
                            {
                                "error": f"Timed out waiting for completion after {int(blocking_timeout_s)}s",
                            },
                        )
                        return
                    continue
            else:
                item = queue.get()
            if item is None:
                break
            gen_text = item.get("text", "")
            if has_tool_calling and gen_text == tool_call_start:
                in_tool_call = True
            elif in_tool_call:
                if gen_text == tool_call_end:
                    tool_calls.append(tool_text)
                    tool_text = ""
                    in_tool_call = False
                else:
                    tool_text += gen_text
            else:
                full_text += gen_text
            finish_reason = item.get("finish_reason")
            prompt_toks = item.get("prompt_tokens", 0)
            gen_toks = item.get("generation_tokens", 0)

        clean_text, extracted = extract_tool_calls(full_text, tools)
        if emit_initial_think and not clean_text.lstrip().startswith("<think>"):
            clean_text = "<think>\n" + clean_text
        all_tool_calls = tool_calls + extracted
        tool_calls_payload = build_tool_calls_payload(all_tool_calls) if all_tool_calls else None
        if tool_calls_payload and finish_reason == "stop":
            finish_reason = "tool_calls"

        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": clean_text, "tool_calls": tool_calls_payload},
                "finish_reason": finish_reason or "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_toks,
                "completion_tokens": gen_toks,
                "total_tokens": prompt_toks + gen_toks,
            },
        }
        self._json_response(200, response)

    def _handle_text(self):
        body = self._parse_body()
        prompt = body.get("prompt", "")
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens", self.args.max_tokens)
        temperature = body.get("temperature", self.args.temperature)
        top_p = body.get("top_p", body.get("topP", self.args.top_p))
        top_k = body.get("top_k", body.get("topK", self.args.top_k))
        repetition_penalty = body.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)
        repetition_context_size = body.get(
            "repetition_context_size", DEFAULT_REPETITION_CONTEXT_SIZE
        )
        seed = body.get("seed", None)
        stop_words = body.get("stop") or []
        model = body.get("model", self.args.model)
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            temperature = float(self.args.temperature)
        try:
            top_p = float(top_p)
        except (TypeError, ValueError):
            top_p = float(self.args.top_p)
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = int(self.args.top_k)
        try:
            repetition_penalty = float(repetition_penalty)
        except (TypeError, ValueError):
            repetition_penalty = DEFAULT_REPETITION_PENALTY
        try:
            repetition_context_size = int(repetition_context_size)
        except (TypeError, ValueError):
            repetition_context_size = DEFAULT_REPETITION_CONTEXT_SIZE
        if temperature < 0:
            temperature = float(self.args.temperature)
        if top_p < 0 or top_p > 1:
            top_p = float(self.args.top_p)
        if top_k < 0:
            top_k = int(self.args.top_k)
        if repetition_penalty < 0:
            repetition_penalty = DEFAULT_REPETITION_PENALTY
        if repetition_context_size < 0:
            repetition_context_size = DEFAULT_REPETITION_CONTEXT_SIZE
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                seed = None
        if seed is None:
            seed = int(time.time_ns() & 0x7FFFFFFF)
        if isinstance(stop_words, str):
            stop_words = [stop_words]
        if not isinstance(stop_words, list):
            stop_words = []
        stop_words = [s for s in stop_words if isinstance(s, str) and s]
        stop_token_sequences = []
        for sw in stop_words[:MAX_STOP_SEQUENCES]:
            try:
                seq = self.tokenizer.encode(sw, add_special_tokens=False)
            except TypeError:
                seq = self.tokenizer.encode(sw)
            if seq:
                stop_token_sequences.append(seq)

        prompt_tokens = self.tokenizer.encode(prompt)

        response_queue = Queue()
        self.dist_state.request_queue.put({
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "repetition_context_size": repetition_context_size,
            "stop_token_sequences": stop_token_sequences,
            "response_queue": response_queue,
            "tools": None,
        })

        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"

        if stream:
            self._stream_text(response_queue, request_id, model)
        else:
            self._blocking_text(response_queue, request_id, model)

    def _stream_text(self, queue, request_id, model):
        self._stream_response()
        try:
            self.wfile.write(b": keepalive\n\n")
            self.wfile.flush()
        except Exception:
            pass

        try:
            while True:
                try:
                    item = queue.get(timeout=10)
                except Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue

                if item is None:
                    break

                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "text": item.get("text", ""), "finish_reason": item.get("finish_reason")}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _blocking_text(self, queue, request_id, model):
        full_text = ""
        finish_reason = None
        prompt_toks = 0
        gen_toks = 0

        blocking_timeout_s = float(os.environ.get("DISTRIBUTED_BLOCKING_TIMEOUT_S", "3600"))
        blocking_poll_s = float(os.environ.get("DISTRIBUTED_BLOCKING_POLL_S", "1"))
        start_t = time.perf_counter()

        while True:
            if blocking_timeout_s > 0:
                try:
                    item = queue.get(timeout=blocking_poll_s)
                except Empty:
                    if (time.perf_counter() - start_t) >= blocking_timeout_s:
                        logging.error(
                            "Blocking text request timed out after %.1fs (id=%s model=%s)",
                            blocking_timeout_s,
                            request_id,
                            model,
                        )
                        self._json_response(
                            504,
                            {
                                "error": f"Timed out waiting for completion after {int(blocking_timeout_s)}s",
                            },
                        )
                        return
                    continue
            else:
                item = queue.get()
            if item is None:
                break
            full_text += item.get("text", "")
            finish_reason = item.get("finish_reason")
            prompt_toks = item.get("prompt_tokens", 0)
            gen_toks = item.get("generation_tokens", 0)

        response = {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "text": full_text, "finish_reason": finish_reason or "stop"}],
            "usage": {"prompt_tokens": prompt_toks, "completion_tokens": gen_toks, "total_tokens": prompt_toks + gen_toks},
        }
        self._json_response(200, response)

    def _handle_anthropic(self):
        body = self._parse_body()
        messages = convert_anthropic_to_openai_messages(body)
        tools = convert_anthropic_tools(body.get("tools"))

        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens", self.args.max_tokens)
        temperature = body.get("temperature", self.args.temperature)
        top_p = body.get("top_p", body.get("topP", self.args.top_p))
        top_k = body.get("top_k", body.get("topK", self.args.top_k))
        repetition_penalty = body.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)
        repetition_context_size = body.get(
            "repetition_context_size", DEFAULT_REPETITION_CONTEXT_SIZE
        )
        seed = body.get("seed", None)
        stop_words = body.get("stop_sequences") or []
        model = body.get("model", self.args.model)
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            temperature = float(self.args.temperature)
        try:
            top_p = float(top_p)
        except (TypeError, ValueError):
            top_p = float(self.args.top_p)
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = int(self.args.top_k)
        try:
            repetition_penalty = float(repetition_penalty)
        except (TypeError, ValueError):
            repetition_penalty = DEFAULT_REPETITION_PENALTY
        try:
            repetition_context_size = int(repetition_context_size)
        except (TypeError, ValueError):
            repetition_context_size = DEFAULT_REPETITION_CONTEXT_SIZE
        if temperature < 0:
            temperature = float(self.args.temperature)
        if top_p < 0 or top_p > 1:
            top_p = float(self.args.top_p)
        if top_k < 0:
            top_k = int(self.args.top_k)
        if repetition_penalty < 0:
            repetition_penalty = DEFAULT_REPETITION_PENALTY
        if repetition_context_size < 0:
            repetition_context_size = DEFAULT_REPETITION_CONTEXT_SIZE
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                seed = None
        if seed is None:
            seed = int(time.time_ns() & 0x7FFFFFFF)
        if isinstance(stop_words, str):
            stop_words = [stop_words]
        if not isinstance(stop_words, list):
            stop_words = []
        stop_words = [s for s in stop_words if isinstance(s, str) and s]
        stop_token_sequences = []
        for sw in stop_words[:MAX_STOP_SEQUENCES]:
            try:
                seq = self.tokenizer.encode(sw, add_special_tokens=False)
            except TypeError:
                seq = self.tokenizer.encode(sw)
            if seq:
                stop_token_sequences.append(seq)

        process_message_content(messages)
        prompt = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        emit_initial_think = prompt.rstrip().endswith("<think>")
        prompt_tokens = self.tokenizer.encode(prompt)

        response_queue = Queue()
        self.dist_state.request_queue.put({
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "repetition_context_size": repetition_context_size,
            "stop_token_sequences": stop_token_sequences,
            "response_queue": response_queue,
            "tools": tools,
        })

        request_id = f"msg_{uuid.uuid4().hex[:24]}"

        if stream:
            self._stream_anthropic(response_queue, request_id, model, tools, emit_initial_think)
        else:
            self._blocking_anthropic(response_queue, request_id, model, tools, emit_initial_think)

    def _stream_anthropic(self, queue, request_id, model, tools, emit_initial_think: bool = False):
        self._stream_response()

        full_text = ""
        tool_calls = []
        tool_text = ""
        in_tool_call = False
        has_tool_calling = getattr(self.tokenizer, "has_tool_calling", False)
        tool_call_start = getattr(self.tokenizer, "tool_call_start", None)
        tool_call_end = getattr(self.tokenizer, "tool_call_end", None)
        finish_reason = None
        prompt_toks = 0
        gen_toks = 0

        try:
            while True:
                try:
                    item = queue.get(timeout=60)
                except Empty:
                    continue
                if item is None:
                    break

                gen_text = item.get("text", "")
                if has_tool_calling and gen_text == tool_call_start:
                    in_tool_call = True
                elif in_tool_call:
                    if gen_text == tool_call_end:
                        tool_calls.append(tool_text)
                        tool_text = ""
                        in_tool_call = False
                    else:
                        tool_text += gen_text
                else:
                    full_text += gen_text

                finish_reason = item.get("finish_reason")
                prompt_toks = item.get("prompt_tokens", prompt_toks)
                gen_toks = item.get("generation_tokens", gen_toks)

            self._send_anthropic_stream_events(
                full_text,
                finish_reason,
                tool_calls,
                prompt_toks,
                gen_toks,
                tools,
                request_id,
                model,
                emit_initial_think,
            )
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_anthropic_stream_events(
        self,
        text: str,
        finish_reason: Optional[str],
        tool_calls: Optional[List[str]],
        input_tokens: int,
        output_tokens: int,
        tools: Optional[List[dict]],
        request_id: str,
        model: str,
        emit_initial_think: bool = False,
    ):
        clean_text, extracted = extract_tool_calls(text, tools)
        if emit_initial_think and not clean_text.lstrip().startswith("<think>"):
            clean_text = "<think>\n" + clean_text
        all_tool_calls = (tool_calls or []) + extracted

        if all_tool_calls:
            finish_reason = "tool_calls"

        self.wfile.write(f"event: message_start\ndata: {json.dumps({
            'type': 'message_start',
            'message': {
                'id': request_id,
                'type': 'message',
                'role': 'assistant',
                'model': model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {'input_tokens': input_tokens, 'output_tokens': 0},
            },
        })}\n\n".encode())

        self.wfile.write(f"event: content_block_start\ndata: {json.dumps({
            'type': 'content_block_start',
            'index': 0,
            'content_block': {'type': 'text', 'text': ''},
        })}\n\n".encode())

        if clean_text:
            self.wfile.write(f"event: content_block_delta\ndata: {json.dumps({
                'type': 'content_block_delta',
                'index': 0,
                'delta': {'type': 'text_delta', 'text': clean_text},
            })}\n\n".encode())

        self.wfile.write(f"event: content_block_stop\ndata: {json.dumps({
            'type': 'content_block_stop', 'index': 0
        })}\n\n".encode())

        block_index = 1
        for tool_data in normalize_tool_calls(all_tool_calls):
            tool_id = f"toolu_{uuid.uuid4().hex[:24]}"

            self.wfile.write(f"event: content_block_start\ndata: {json.dumps({
                'type': 'content_block_start',
                'index': block_index,
                'content_block': {
                    'type': 'tool_use',
                    'id': tool_id,
                    'name': tool_data.get('name', ''),
                    'input': {},
                },
            })}\n\n".encode())

            self.wfile.write(f"event: content_block_delta\ndata: {json.dumps({
                'type': 'content_block_delta',
                'index': block_index,
                'delta': {
                    'type': 'input_json_delta',
                    'partial_json': json.dumps(tool_data.get('arguments', {})),
                },
            })}\n\n".encode())

            self.wfile.write(f"event: content_block_stop\ndata: {json.dumps({
                'type': 'content_block_stop', 'index': block_index
            })}\n\n".encode())
            block_index += 1

        stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
        self.wfile.write(f"event: message_delta\ndata: {json.dumps({
            'type': 'message_delta',
            'delta': {'stop_reason': stop_reason_map.get(finish_reason, finish_reason), 'stop_sequence': None},
            'usage': {'output_tokens': output_tokens},
        })}\n\n".encode())

        self.wfile.write(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode())
        self.wfile.flush()

    def _blocking_anthropic(self, queue, request_id, model, tools, emit_initial_think: bool = False):
        full_text = ""
        tool_calls = []
        tool_text = ""
        in_tool_call = False
        has_tool_calling = getattr(self.tokenizer, "has_tool_calling", False)
        tool_call_start = getattr(self.tokenizer, "tool_call_start", None)
        tool_call_end = getattr(self.tokenizer, "tool_call_end", None)
        finish_reason = None
        prompt_toks = 0
        gen_toks = 0

        while True:
            try:
                item = queue.get(timeout=120)
            except Empty:
                break
            if item is None:
                break
            gen_text = item.get("text", "")
            if has_tool_calling and gen_text == tool_call_start:
                in_tool_call = True
            elif in_tool_call:
                if gen_text == tool_call_end:
                    tool_calls.append(tool_text)
                    tool_text = ""
                    in_tool_call = False
                else:
                    tool_text += gen_text
            else:
                full_text += gen_text
            finish_reason = item.get("finish_reason")
            prompt_toks = item.get("prompt_tokens", 0)
            gen_toks = item.get("generation_tokens", 0)

        clean_text, extracted = extract_tool_calls(full_text, tools)
        if emit_initial_think and not clean_text.lstrip().startswith("<think>"):
            clean_text = "<think>\n" + clean_text
        content = []

        if clean_text:
            content.append({"type": "text", "text": clean_text})

        all_tool_calls = tool_calls + extracted
        for td in normalize_tool_calls(all_tool_calls):
            content.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": td.get("name", ""),
                "input": td.get("arguments", {}),
            })

        has_tool = any(b.get("type") == "tool_use" for b in content)
        stop_reason = "tool_use" if has_tool else ("end_turn" if finish_reason == "stop" else "max_tokens")

        response = {
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": prompt_toks, "output_tokens": gen_toks},
        }
        self._json_response(200, response)


def generation_loop(dist_state, model, tokenizer, args, prompt_cache_store=None):
    """Main generation loop running on ALL ranks.

    All ranks must execute this together for pipeline parallelism to work.
    """
    rank = dist_state.rank
    logging.info(f"Generation loop started on rank {rank}")
    
    # Initialize prompt cache store if not provided
    if prompt_cache_store is None:
        prompt_cache_store = LRUPromptCache(max_size=max(0, int(args.prompt_cache_size)))

    request_n = 0

    while True:
        # All ranks sync and check for requests
        (
            prompt_tokens,
            max_tokens,
            seed,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            repetition_context_size,
            stop_token_sequences,
            response_queue,
            request,
        ) = dist_state.broadcast_request()

        if prompt_tokens is None:
            # No request - brief sleep to avoid busy-waiting
            time.sleep(0.005)
            continue

        mx.random.seed(int(seed))
        request_n += 1

        if rank == 0:
            logging.info(
                "Request params: req=%d seed=%d max_tokens=%d temperature=%.3f top_p=%.3f top_k=%d repetition_penalty=%.3f repetition_context_size=%d stop_sequences=%d",
                request_n,
                int(seed),
                int(max_tokens) if max_tokens is not None else 0,
                float(temperature),
                float(top_p),
                int(top_k),
                float(repetition_penalty),
                int(repetition_context_size),
                len(stop_token_sequences or []),
            )

        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        logits_processors = None
        if repetition_penalty and repetition_penalty != 0.0:
            logits_processors = make_logits_processors(
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )
            if rank == 0:
                logging.info(
                    "Using repetition_penalty=%.3f repetition_context_size=%d",
                    repetition_penalty,
                    repetition_context_size,
                )

        # All ranks generate together
        # Try to fetch from prompt cache
        cached_prompt_cache, tokens_to_process = prompt_cache_store.fetch_nearest_cache(
            args.model, prompt_tokens
        )
        full_prompt_len = len(prompt_tokens)

        if rank == 0:
            cache_hit = cached_prompt_cache is not None
            logging.info(f"Starting generation: prompt_len={len(prompt_tokens)}, cache_hit={cache_hit}, max_tokens={max_tokens}")
            gen_start_t = time.perf_counter()
            first_token_dt = None
        else:
            gen_start_t = None
            first_token_dt = None
        
        # Create prompt cache if not found
        if cached_prompt_cache is None:
            prompt_cache = make_prompt_cache(model)
        else:
            prompt_cache = cached_prompt_cache

        # If we have an exact cache match, tokens_to_process can be empty. MLX
        # generation requires a non-empty prompt (unless using input embeddings),
        # so ensure we always process at least one token.
        if not tokens_to_process:
            if not prompt_tokens:
                if rank == 0 and response_queue is not None:
                    response_queue.put(
                        {
                            "text": "Error: empty prompt_tokens (cannot generate).",
                            "finish_reason": "error",
                        }
                    )
                    response_queue.put(None)
                continue

            if can_trim_prompt_cache(prompt_cache):
                # Trim one token off the cache and re-process it to seed generation.
                try:
                    trim_prompt_cache(prompt_cache, 1)
                    tokens_to_process = [prompt_tokens[-1]]
                    if rank == 0:
                        logging.debug(
                            "Exact prompt cache hit; trimmed 1 token to avoid empty prompt."
                        )
                except Exception:
                    if rank == 0:
                        logging.debug(
                            "Failed to trim prompt cache for exact match; rebuilding cache.",
                            exc_info=True,
                        )
                    prompt_cache = make_prompt_cache(model)
                    tokens_to_process = prompt_tokens
            else:
                # Fallback: rebuild prompt cache and re-process the full prompt.
                prompt_cache = make_prompt_cache(model)
                tokens_to_process = prompt_tokens

        prompt = mx.array(tokens_to_process, dtype=mx.int32)
        cache_key = prompt_tokens[:]

        # Stop-sequence buffering (rank 0 only) to avoid emitting partial stop strings.
        pending_items = (
            deque() if rank == 0 and response_queue is not None else None
        )

        stop_sequences = [s for s in (stop_token_sequences or []) if s]
        stop_lps = [build_kmp_lps(s) for s in stop_sequences]
        stop_match = [0] * len(stop_sequences)

        last_response = None
        try:
            for response in stream_generate(
                model, tokenizer, prompt, 
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
                sampler=sampler,
                logits_processors=logits_processors,
            ):
                last_response = response
                cache_key.append(response.token)
                if rank == 0 and response_queue is not None and first_token_dt is None:
                    first_token_dt = time.perf_counter() - gen_start_t
                    try:
                        prompt_tps = float(getattr(response, "prompt_tps", 0.0) or 0.0)
                    except Exception:
                        prompt_tps = 0.0
                    logging.info(
                        "First token: dt=%.3fs prompt_suffix=%d full_prompt_len=%d prompt_tps=%.3f",
                        first_token_dt,
                        int(getattr(response, "prompt_tokens", 0) or 0),
                        full_prompt_len,
                        prompt_tps,
                    )
                holdback = 0
                stop_trim = 0
                if stop_sequences:
                    tok = int(response.token)
                    for i, seq in enumerate(stop_sequences):
                        l = stop_match[i]
                        while l > 0 and seq[l] != tok:
                            l = stop_lps[i][l - 1]
                        if l < len(seq) and seq[l] == tok:
                            l += 1
                        if l > holdback:
                            holdback = l
                        if l == len(seq) and len(seq) > stop_trim:
                            stop_trim = len(seq)
                        stop_match[i] = l

                item = None
                if rank == 0 and response_queue is not None:
                    item = {
                        "text": response.text,
                        "finish_reason": response.finish_reason,
                        # stream_generate reports prompt_tokens as the length of the prompt
                        # it was called with (which may be only the non-cached suffix).
                        # For OpenAI usage, report the full prompt length.
                        "prompt_tokens": full_prompt_len,
                        "generation_tokens": response.generation_tokens,
                        "token": response.token,
                    }
                    pending_items.append(item)

                # Stop early if we hit a stop sequence (discard the stop sequence tokens).
                if stop_trim > 0:
                    if pending_items is not None:
                        for _ in range(min(stop_trim, len(pending_items))):
                            pending_items.pop()
                    break

                # Flush buffered items except the current stop-prefix holdback.
                if pending_items is not None and holdback >= 0:
                    flush_count = len(pending_items) - holdback
                    for _ in range(max(0, flush_count)):
                        response_queue.put(pending_items.popleft())

            # Flush anything left (e.g. overlap prefixes when generation ended).
            if pending_items is not None:
                while pending_items:
                    response_queue.put(pending_items.popleft())

            if rank == 0 and response_queue is not None:
                response_queue.put(None)

            if rank == 0:
                total_dt = time.perf_counter() - gen_start_t
                gen_tokens = int(getattr(last_response, "generation_tokens", 0) or 0) if last_response is not None else 0
                logging.info(
                    "Generation finished: gen_tokens=%d total_dt=%.3fs first_token_dt=%.3fs",
                    gen_tokens,
                    total_dt,
                    first_token_dt if first_token_dt is not None else 0.0,
                )

            # Save prompt-only cache for future requests (better reuse with tool calls)
            generated_count = max(0, len(cache_key) - len(prompt_tokens))
            if generated_count > 0 and can_trim_prompt_cache(prompt_cache):
                trim_prompt_cache(prompt_cache, generated_count)
                prompt_cache_store.insert_cache(args.model, prompt_tokens, prompt_cache)
            else:
                prompt_cache_store.insert_cache(args.model, cache_key, prompt_cache)

        except Exception as e:
            logging.exception(f"Generation error on rank {rank}")
            if rank == 0 and response_queue is not None:
                response_queue.put({"text": f"Error: {e}", "finish_reason": "error"})
                response_queue.put(None)


def run_http_server(dist_state, tokenizer, args):
    """Run HTTP server (rank 0 only)."""
    def factory(*a, **kw):
        return DistributedHandler(dist_state, tokenizer, args, *a, **kw)

    addr = (args.host, args.port)
    infos = socket.getaddrinfo(*addr, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    ThreadingHTTPServer.address_family = next(iter(infos))[0]

    server = ThreadingHTTPServer(addr, factory)
    logging.info(f"HTTP server on {args.host}:{args.port}")

    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="MLX Distributed HTTP Server")
    parser.add_argument("--model", type=str, required=True, help="Model path or HuggingFace repo")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--chat-template", type=str, default="")
    parser.add_argument("--use-default-chat-template", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--prompt-cache-size",
        type=int,
        default=2,
        help="Maximum number of prompt-cache entries to keep (LRU). Set to 0 to disable.",
    )

    args = parser.parse_args()

    # Initialize distributed
    group = mx.distributed.init()
    rank = group.rank()
    world_size = group.size()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=f"%(asctime)s - [Rank {rank}] %(message)s",
    )

    logging.info(f"Distributed: rank {rank}/{world_size}")

    # Load model with sharding
    logging.info(f"Loading {args.model} (pipeline parallelism)")
    model, tokenizer = sharded_load(args.model, pipeline_group=group, tensor_group=None)
    logging.info("Model loaded")

    if args.use_default_chat_template and tokenizer.chat_template is None:
        tokenizer.chat_template = tokenizer.default_chat_template
    if args.chat_template:
        tokenizer.chat_template = args.chat_template

    dist_state = DistributedState(group)

    if rank == 0:
        # Rank 0: HTTP server in background, generation loop in foreground
        http_thread = Thread(target=run_http_server, args=(dist_state, tokenizer, args), daemon=True)
        http_thread.start()
        warnings.warn("Distributed server: not recommended for production")

    # All ranks run generation loop
    generation_loop(dist_state, model, tokenizer, args)


if __name__ == "__main__":
    main()
