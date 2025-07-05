# Copyright © 2025 Apple Inc.

import argparse
import copy
import time
import types
import importlib
from typing import Tuple, Type, Optional, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
import numpy as np
from mlx.utils import tree_map
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset as hf_load_dataset
import random

from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.trainer import iterate_batches
from mlx_lm.tuner.utils import print_trainable_parameters
from mlx_lm.utils import (
    load_model,
    get_model_path,
    quantize_model,
    save,
)


class Catcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def __call__(self, *args, **kwargs):
        self.outputs = self.module(*args, **kwargs)
        return self.outputs


def get_t5_encoder_classes(config: dict) -> Tuple[Type[nn.Module], Type]:
    """Get T5 encoder model classes for custom loading"""
    model_type = config.get("model_type", "").lower()
    
    if model_type not in ["t5"]:
        raise ValueError(f"This loader only supports T5 models, got: {model_type}")
    
    # Import the T5 encoder module
    try:
        t5_module = importlib.import_module("mlx_lm.models.t5_encoder")
    except ImportError:
        raise ImportError(
            "Could not import mlx_lm.models.t5_encoder. "
            "Ensure the T5 encoder model is available."
        )
    
    return t5_module.Model, t5_module.ModelArgs


def load_t5_encoder(
    model_path: str,
) -> Tuple[nn.Module, Any]:
    """Load T5 encoder model using custom loading pattern"""
    # Get the model path
    model_path_resolved = get_model_path(model_path)
    
    # Load model with custom T5 encoder classes
    model, _ = load_model(
        model_path=model_path_resolved,
        get_model_classes=get_t5_encoder_classes,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True
    )
    
    return model, tokenizer


def dwq_t5_quantize(
    model,
    q_model,
    opt,
    data,
    batch_size: int = 2,
    max_seq_length: int = 512,
    activation_layer_step: float = 0.25,
    activation_loss_weight: float = 1.0,
    dtype: mx.Dtype = mx.bfloat16,
):
    group = mx.distributed.init()
    world_size = group.size()
    rank = group.rank()

    def unfreeze(_, m):
        if hasattr(m, "bits") and hasattr(m, "group_size"):
            m.unfreeze(keys=["scales", "biases"], recurse=False)

    q_model.apply_to_modules(unfreeze)
    print_trainable_parameters(q_model)

    # For T5 encoder, we target encoder layers
    encoder_layers = model.encoder.layers if hasattr(model, 'encoder') else model.layers
    q_encoder_layers = q_model.encoder.layers if hasattr(q_model, 'encoder') else q_model.layers
    
    layer_id_step = max(int(activation_layer_step * len(encoder_layers)), 1)
    layer_ids = list(range(len(encoder_layers)))[layer_id_step::layer_id_step]

    for lid in layer_ids:
        encoder_layers[lid] = Catcher(encoder_layers[lid])
        q_encoder_layers[lid] = Catcher(q_encoder_layers[lid])

    def forward(model, inputs):
        # T5 encoder returns hidden states directly, not logits
        hidden_states = model(inputs).astype(mx.float32)
        extra_targets = [
            encoder_layers[lid].outputs.astype(mx.float32) for lid in layer_ids
        ]
        for lid in layer_ids:
            encoder_layers[lid].outputs = None
        return hidden_states, extra_targets

    def loss_fn(params, x, targets, extra_targets, lengths):
        q_model.update(tree_map(lambda x: x.astype(dtype), params))
        hidden_states, q_extra_targets = forward(q_model, x)
        
        # MSE loss on hidden states instead of KL divergence
        mask = mx.arange(1, 1 + targets.shape[1]) < lengths[:, None]
        num_tokens = mask.sum()
        
        # Hidden state loss
        hidden_loss = nn.losses.mse_loss(hidden_states, targets, reduction="none")
        masked_hidden_loss = (mask[..., None] * hidden_loss).sum() / num_tokens
        
        # Activation loss for intermediate layers
        act_loss = mx.stack(
            [
                (mask[..., None] * nn.losses.mse_loss(qe, e, reduction="none")).sum() / num_tokens
                for qe, e in zip(q_extra_targets, extra_targets)
            ]
        )
        
        loss = masked_hidden_loss + activation_loss_weight * act_loss.mean()
        return loss, num_tokens

    def step(inputs, targets, extra_targets, lengths, params):
        (loss, num_tokens), grads = mx.value_and_grad(loss_fn)(
            params, inputs, targets, extra_targets, lengths
        )
        grads = nn.average_gradients(grads)
        params = opt.apply_gradients(grads, params)
        return loss, num_tokens, params

    # Accumulate learned weights in higher precision
    params = tree_map(
        lambda x: x.astype(mx.float32),
        q_model.trainable_parameters(),
    )

    total_loss = 0.0
    total_tokens = 0
    tokens = 0
    tic = time.time()
    for it, (batch, lengths) in (
        progress_bar := tqdm(
            enumerate(iterate_batches(data, batch_size, max_seq_length)),
            total=len(data) // batch_size,
        )
    ):
        # No token shifting for T5 encoder - use full sequence
        targets, extra_targets = forward(model, batch)
        mx.eval(targets, extra_targets)
        loss, num_tokens, params = step(batch, targets, extra_targets, lengths, params)
        mx.eval(loss, params)
        loss = mx.distributed.all_sum(loss, stream=mx.cpu).item() / world_size
        num_tokens = mx.distributed.all_sum(num_tokens, stream=mx.cpu).item()
        tokens += num_tokens
        total_loss += loss * num_tokens
        if rank == 0:
            progress_bar.set_description(desc=f"{loss=:.4f}")
            if (it + 1) % 20 == 0:
                tokens_per_sec = tokens / (time.time() - tic)
                peak_memory_gb = mx.get_peak_memory() / 1e9
                avg_loss = total_loss / tokens
                total_tokens += tokens
                tqdm.write(
                    f"{it=}, {avg_loss=:.4f}, {total_tokens=},"
                    f" {tokens_per_sec=:.3f}, {peak_memory_gb=:.3f}",
                )
                tic = time.time()
                tokens = 0
                total_loss = 0
    q_model.update(tree_map(lambda x: x.astype(dtype), params))
    for lid in layer_ids:
        q_encoder_layers[lid] = q_encoder_layers[lid].module


def load_t5_data(tokenizer, data_path: str, num_samples: int, max_seq_length: int):
    """Load data for T5 encoder training - uses full sequences without shifting"""
    args = types.SimpleNamespace(
        hf_dataset={
            "path": data_path,
            "train_split": f"train",
            "valid_split": "train[:1]",
        },
        train=True,
        test=False,
    )
    dataset = load_dataset(args, tokenizer)[0]
    perm = np.random.permutation(len(dataset))[:num_samples].tolist()

    def process(idx):
        tokens, _ = dataset.process(dataset[idx])
        # For T5 encoder, we use the full sequence
        return (tokens[:max_seq_length], len(tokens[:max_seq_length]))

    return [process(i) for i in perm]


def get_msmarco_calibration_data(num_samples=2048):
    """Get diverse samples from MS MARCO for DWQ calibration"""
    # Load MS MARCO passages
    dataset = hf_load_dataset("ms_marco", "v2.1", split="train", streaming=True)
    
    samples = []
    seen_lengths = set()
    
    for item in dataset:
        # Get passage text
        text = item['passages']['passage_text'][0]
        
        # Diversify by length
        length_bucket = len(text) // 100
        if length_bucket not in seen_lengths or len(samples) < num_samples:
            samples.append(text)
            seen_lengths.add(length_bucket)
            
        if len(samples) >= num_samples:
            break
    
    return samples


def get_diverse_calibration_mix(num_samples=2048):
    """Mix different text types for better calibration"""
    samples = []
    
    # 1. Questions (25%)
    try:
        nq = hf_load_dataset("natural_questions", split="train", streaming=True)
        for i, item in enumerate(nq):
            if i >= num_samples // 4:
                break
            samples.append(item['question']['text'])
    except Exception as e:
        print(f"Warning: Could not load Natural Questions: {e}")
    
    # 2. Passages (25%)
    try:
        marco = hf_load_dataset("ms_marco", "v2.1", split="train", streaming=True)
        for i, item in enumerate(marco):
            if i >= num_samples // 4:
                break
            samples.append(item['passages']['passage_text'][0][:512])
    except Exception as e:
        print(f"Warning: Could not load MS MARCO: {e}")
    
    # 3. Sentences (25%)
    try:
        stsb = hf_load_dataset("mteb/stsbenchmark-sts", split="train")
        for i in range(min(len(stsb), num_samples // 4)):
            samples.append(stsb[i]['sentence1'])
    except Exception as e:
        print(f"Warning: Could not load STS-B: {e}")
    
    # 4. Definitions/Descriptions (25%)
    try:
        wiki = hf_load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        for i, item in enumerate(wiki):
            if i >= num_samples // 4:
                break
            # Get first sentence as definition
            text = item['text'].split('.')[0] + '.'
            if 20 < len(text) < 200:  # Good definition length
                samples.append(text)
    except Exception as e:
        print(f"Warning: Could not load Wikipedia: {e}")
    
    random.shuffle(samples)
    return samples[:num_samples]


def load_calibration_data_from_standard_datasets(
    tokenizer, 
    num_samples: int = 2048,
    max_seq_length: int = 512,
    dataset_name: str = "mixed"
):
    """Load calibration data from standard datasets"""
    
    if dataset_name == "msmarco":
        texts = get_msmarco_calibration_data(num_samples)
    elif dataset_name == "stsb":
        dataset = hf_load_dataset("mteb/stsbenchmark-sts", split="train")
        texts = [item['sentence1'] for item in dataset[:num_samples]]
    elif dataset_name == "nq":
        dataset = hf_load_dataset("natural_questions", split="train", streaming=True)
        texts = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            texts.append(item['question']['text'])
    elif dataset_name == "mixed":
        texts = get_diverse_calibration_mix(num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Tokenize
    data = []
    for text in texts:
        tokens = tokenizer.encode(text, max_length=max_seq_length, truncation=True)
        if len(tokens) > 10:  # Skip very short sequences
            data.append((tokens, len(tokens)))
    
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="google/t5-small")
    parser.add_argument("--quantized-model", default=None)
    parser.add_argument(
        "--mlx-path", default="mlx_model", help="Path to save the quantized model."
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Bits per weight for quantization.",
    )
    parser.add_argument(
        "--group-size", type=int, default=64, help="Group size for quantization."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        help="Number of samples to use for training.",
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--data-path",
        type=str,
        default="allenai/tulu-3-sft-mixture",
        help="A Hugging Face dataset which is compatible with an mlx-lm dataset format.",
    )
    parser.add_argument(
        "--activation-layer-step",
        type=float,
        default=0.25,
        help="Fraction of layers to use for activation matching.",
    )
    parser.add_argument(
        "--activation-loss-weight",
        type=float,
        default=1.0,
        help="Weight for activation loss component.",
    )
    parser.add_argument(
        "--calibration-dataset",
        type=str,
        default="mixed",
        choices=["msmarco", "stsb", "nq", "mixed", "custom"],
        help="Standard dataset to use for calibration (default: mixed)"
    )
    args = parser.parse_args()

    group = mx.distributed.init()

    num_samples = args.num_samples
    if num_samples % group.size() > 0:
        num_samples += group.size() - num_samples % group.size()

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # Load T5 encoder model using custom loading
    model, tokenizer = load_t5_encoder(args.model)
    
    # Load calibration data
    if args.calibration_dataset == "custom":
        calibration_data = load_t5_data(
            tokenizer, args.data_path, args.num_samples, args.max_seq_length
        )
    else:
        calibration_data = load_calibration_data_from_standard_datasets(
            tokenizer, 
            args.num_samples, 
            args.max_seq_length,
            args.calibration_dataset
        )

    if args.quantized_model is not None:
        q_model, _ = load_t5_encoder(args.quantized_model)
    else:
        q_model = copy.deepcopy(model)
        # Get config for quantization
        config_dict = model.config.__dict__ if hasattr(model, 'config') else {}
        _, config = quantize_model(
            q_model,
            config_dict,
            q_group_size=args.group_size,
            q_bits=args.bits,
        )

    opt = optimizers.Adam(learning_rate=args.learning_rate, bias_correction=True)
    dwq_t5_quantize(
        model,
        q_model,
        opt,
        calibration_data,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        activation_layer_step=args.activation_layer_step,
        activation_loss_weight=args.activation_loss_weight,
    )
    # Save quantized model with proper weight mapping
    # First unsanitize weights back to HuggingFace format
    model_weights = dict(q_model.named_parameters())
    if hasattr(q_model, 'unsanitize'):
        model_weights = q_model.unsanitize(model_weights)
    
    # Update model with unsanitized weights for saving
    q_model.update(model_weights)
    
    save(
        args.mlx_path,
        args.model,
        q_model,
        tokenizer,
        config,
    )


if __name__ == "__main__":
    main()