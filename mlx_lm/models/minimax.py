# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, List, Optional

import logging
import mlx.core as mx
import os
import time
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .switch_layers import SwitchGLU

try:
    from mlx.nn.layers.distributed import shard_linear, shard_inplace, sum_gradients
except ImportError:
    shard_linear = None
    shard_inplace = None
    sum_gradients = None


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    num_experts_per_tok: int
    num_local_experts: int
    shared_intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    vocab_size: int
    tie_word_embeddings: bool = False
    scoring_func: str = "sigmoid"
    head_dim: Optional[int] = None
    use_qk_norm: bool = True


class MiniMaxAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_dim = hidden_size = args.hidden_size

        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = head_dim = (
            args.head_dim or hidden_size // args.num_attention_heads
        )
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * head_dim, args.hidden_size, bias=False
        )

        self.use_qk_norm = args.use_qk_norm if hasattr(args, "use_qk_norm") else False
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(
                head_dim * self.num_attention_heads, eps=args.rms_norm_eps
            )
            self.k_norm = nn.RMSNorm(
                head_dim * self.num_key_value_heads, eps=args.rms_norm_eps
            )

        self.rope = nn.RoPE(args.rotary_dim, traditional=False, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)


class MiniMaxSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.num_experts = args.num_local_experts

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_local_experts
        )
        self.e_score_correction_bias = mx.zeros((args.num_local_experts,))

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x.astype(mx.float32))

        scores = mx.sigmoid(gates)
        orig_scores = scores
        scores = scores + self.e_score_correction_bias

        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(orig_scores, inds, axis=-1)

        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        scores = scores.astype(x.dtype)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        
        return y


class MiniMaxDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.self_attn = MiniMaxAttention(args)

        self.block_sparse_moe = MiniMaxSparseMoeBlock(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = x + self.self_attn(self.input_layernorm(x), mask, cache)
        r = r + self.block_sparse_moe(self.post_attention_layernorm(r))
        return r


from .pipeline import PipelineMixin


class MiniMaxModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        PipelineMixin.__init__(self)
        nn.Module.__init__(self)
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = [
            MiniMaxDecoderLayer(args=args) 
            for _ in range(args.num_hidden_layers)
        ]

        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def pipeline(self, group):
        split = os.environ.get("MINIMAX_PIPELINE_SPLIT")
        if not split:
            return PipelineMixin.pipeline(self, group)

        counts = [int(x.strip()) for x in split.split(",") if x.strip()]
        if len(counts) != group.size():
            raise ValueError(
                f"MINIMAX_PIPELINE_SPLIT expects {group.size()} entries but got {len(counts)}"
            )
        total_layers = len(self.layers)
        if sum(counts) != total_layers:
            raise ValueError(
                f"MINIMAX_PIPELINE_SPLIT must sum to {total_layers} layers but got {sum(counts)}"
            )

        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()

        # Rank 0 is the last stage; rank N-1 is the first stage.
        prefix = sum(counts[: self.pipeline_rank])
        self.end_idx = total_layers - prefix
        self.start_idx = self.end_idx - counts[self.pipeline_rank]

        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        trace = os.environ.get("MINIMAX_PIPELINE_TRACE") == "1"
        if trace:
            limit = int(os.environ.get("MINIMAX_PIPELINE_TRACE_LIMIT", "4") or 4)
            call_n = getattr(self, "_trace_call_n", 0)
            setattr(self, "_trace_call_n", call_n + 1)
            trace = call_n < limit

        if trace:
            t0 = time.perf_counter()
            logging.info(
                "minimax: fwd begin rank=%d/%d inputs=%s start=%d end=%d",
                getattr(self, "pipeline_rank", -1),
                getattr(self, "pipeline_size", -1),
                tuple(inputs.shape),
                getattr(self, "start_idx", -1),
                getattr(self, "end_idx", -1),
            )

        # Always embed tokens first (needed for mask computation)
        h = self.embed_tokens(inputs)
        if trace:
            logging.info("minimax: embed done dt=%.3fs", time.perf_counter() - t0)
        mx.eval(h)  # Sync point for pipeline
        if trace:
            logging.info("minimax: embed eval done dt=%.3fs", time.perf_counter() - t0)

        # Get the number of actual layers on this rank
        local_layers = self.pipeline_layers  # Uses start_idx:end_idx
        num_local_layers = len(local_layers)

        if cache is None:
            cache = [None] * num_local_layers

        mask = create_attention_mask(h, cache[0] if cache else None)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        # Receive hidden states from previous rank in pipeline
        if pipeline_size > 1 and pipeline_rank < pipeline_size - 1:
            if trace:
                logging.info(
                    "minimax: recv_like begin dt=%.3fs src=%d",
                    time.perf_counter() - t0,
                    pipeline_rank + 1,
                )
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))
            mx.eval(h)  # Sync point: ensure recv completes
            if trace:
                logging.info("minimax: recv_like done dt=%.3fs", time.perf_counter() - t0)

        # Process layers for this rank
        for i, layer in enumerate(local_layers):
            layer_cache = cache[i] if i < len(cache) else None
            h = layer(h, mask, layer_cache)
        mx.eval(h)  # Sync point: ensure layers complete
        if trace:
            logging.info("minimax: layers done dt=%.3fs", time.perf_counter() - t0)

        # Send to next rank in pipeline
        if pipeline_size > 1 and pipeline_rank != 0:
            if trace:
                logging.info(
                    "minimax: send begin dt=%.3fs dst=%d",
                    time.perf_counter() - t0,
                    (pipeline_rank - 1) % pipeline_size,
                )
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            mx.eval(h)  # Sync point: ensure send completes
            if trace:
                logging.info("minimax: send done dt=%.3fs", time.perf_counter() - t0)

        # Broadcast final hidden states (rank 0) to all ranks.
        #
        # all_gather is conceptually fine here, but we've observed it can hang on
        # multi-machine ring runs. Since we only need rank 0's output, use an
        # all_sum-based broadcast instead: rank 0 contributes `h`, other ranks
        # contribute zeros.
        if pipeline_size > 1:
            if trace:
                logging.info("minimax: broadcast(all_sum) begin dt=%.3fs", time.perf_counter() - t0)
            h_sum = h if pipeline_rank == 0 else mx.zeros_like(h)
            h = mx.distributed.all_sum(h_sum)
            mx.eval(h)  # Sync point: ensure broadcast completes
            if trace:
                logging.info("minimax: broadcast(all_sum) done dt=%.3fs", time.perf_counter() - t0)

        result = self.norm(h)
        mx.eval(result)  # Sync point: ensure norm completes
        if trace:
            logging.info("minimax: norm done dt=%.3fs", time.perf_counter() - t0)
        
        return result


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniMaxModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs=inputs, mask=mask, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        """Dequantize FP8 weights and restructure MoE experts."""

        def dequant(weight, scale_inv):
            dtype = weight.dtype
            bs = 128  # block size
            m, n = weight.shape
            pad_bottom = (-m) % bs
            pad_side = (-n) % bs
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(
                ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
            )
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_bottom, n + pad_side
            )
            return weight[:m, :n].astype(dtype)

        # Dequantize
        new_weights = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                weight = weights[wk]
                weight = dequant(weight, scale_inv)
                new_weights[wk] = weight
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        # Step 2: Handle MoE expert weights restructuring
        if "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in weights:
            return weights

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            for orig_name, new_name in mapping.items():
                if f"{prefix}.block_sparse_moe.experts.0.{orig_name}.weight" in weights:
                    to_join = [
                        weights.pop(
                            f"{prefix}.block_sparse_moe.experts.{e}.{orig_name}.weight"
                        )
                        for e in range(self.args.num_local_experts)
                    ]
                    weights[
                        f"{prefix}.block_sparse_moe.switch_mlp.{new_name}.weight"
                    ] = mx.stack(to_join)

        return weights

    @property
    def layers(self):
        """Return only the layers for this rank (local layers in pipeline mode)."""
        return self.model.pipeline_layers
    
    def make_cache(self):
        """Create cache for local layers only."""
        from .cache import KVCache
        return [KVCache() for _ in self.layers]

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("block_sparse_moe.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
