# DWQ T5 Encoder Quantization

This document describes how to use Dynamic Weight Quantization (DWQ) for T5 encoder models.

## Overview

DWQ T5 is an adaptation of the DWQ algorithm specifically designed for T5 encoder models. Unlike autoregressive language models, T5 encoders process sequences bidirectionally and output hidden states rather than logits.

## Key Features

- **T5 Encoder Support**: Adapted for bidirectional processing without token shifting
- **MSE Loss**: Uses Mean Squared Error on hidden states instead of KL divergence
- **Activation Matching**: Preserves intermediate layer activations during quantization
- **Custom Model Loading**: Supports local T5 models with proper weight mapping
- **Standard Dataset Support**: Built-in calibration data from MS MARCO, STS-B, and more

## Usage

### Basic Training

```bash
python -m mlx_lm.quant.dwq_t5 --model /path/to/local/t5/model --max-seq-length 512
```

### Advanced Training

```bash
python -m mlx_lm.quant.dwq_t5 \
    --model /path/to/local/t5/model \
    --mlx-path ./quantized_t5_model \
    --bits 4 \
    --group-size 64 \
    --num-samples 1024 \
    --max-seq-length 512 \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --activation-layer-step 0.25 \
    --activation-loss-weight 1.0
```

## Loading Models from Local Path

### For Training (Input Model)

```bash
# Local HuggingFace model directory
--model /Users/username/models/t5-small

# Or relative path
--model ./models/t5-small
```

### For Pre-quantized Model

```bash
# Use existing quantized model as starting point
--model /path/to/original/model \
--quantized-model /path/to/quantized/model
```

## Example Commands

### 1. Train with Local T5-small

```bash
python -m mlx_lm.quant.dwq_t5 \
    --model /Users/jieanchen/models/t5-small \
    --mlx-path ./t5_dwq_quantized \
    --num-samples 512 \
    --max-seq-length 256
```

### 2. Resume from Checkpoint

```bash
python -m mlx_lm.quant.dwq_t5 \
    --model /Users/jieanchen/models/t5-small \
    --quantized-model ./t5_dwq_quantized \
    --mlx-path ./t5_dwq_continued
```

### 3. Use Standard Datasets for Calibration

```bash
# Use MS MARCO for retrieval-focused quantization
python -m mlx_lm.quant.dwq_t5 \
    --model google/t5-small \
    --calibration-dataset msmarco \
    --num-samples 2048

# Use STS-B for similarity-focused quantization
python -m mlx_lm.quant.dwq_t5 \
    --model google/t5-small \
    --calibration-dataset stsb \
    --num-samples 2048

# Use mixed data for general-purpose quantization (default)
python -m mlx_lm.quant.dwq_t5 \
    --model google/t5-small \
    --calibration-dataset mixed \
    --num-samples 1024

# Use Natural Questions for QA-focused quantization
python -m mlx_lm.quant.dwq_t5 \
    --model google/t5-small \
    --calibration-dataset nq \
    --num-samples 2048

# Use custom dataset (original behavior)
python -m mlx_lm.quant.dwq_t5 \
    --model google/t5-small \
    --calibration-dataset custom \
    --data-path allenai/tulu-3-sft-mixture
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `google/t5-small` | Path to the original T5 model |
| `--quantized-model` | `None` | Path to pre-quantized model (optional) |
| `--mlx-path` | `mlx_model` | Output path for quantized model |
| `--bits` | `4` | Bits per weight for quantization |
| `--group-size` | `64` | Group size for quantization |
| `--num-samples` | `1024` | Number of calibration samples |
| `--max-seq-length` | `512` | Maximum sequence length |
| `--batch-size` | `4` | Training batch size |
| `--learning-rate` | `1e-5` | Learning rate for optimization |
| `--activation-layer-step` | `0.25` | Fraction of layers for activation matching |
| `--activation-loss-weight` | `1.0` | Weight for activation loss component |
| `--calibration-dataset` | `mixed` | Standard dataset for calibration: `msmarco`, `stsb`, `nq`, `mixed`, `custom` |
| `--data-path` | `allenai/tulu-3-sft-mixture` | Training dataset path (used when `--calibration-dataset custom`) |
| `--seed` | `123` | Random seed |

## Model Structure

The T5 encoder model includes:

- **Embedding Layer**: Token embeddings
- **Encoder Stack**: Multiple transformer encoder layers
- **Relative Position Bias**: T5-specific positional encoding
- **Layer Normalization**: RMSNorm layers

## Loss Function

The DWQ T5 loss combines:

1. **Hidden State Loss**: MSE between original and quantized model outputs
2. **Activation Loss**: MSE between intermediate layer activations
3. **Sequence Masking**: Proper handling of variable-length sequences

```python
loss = hidden_state_loss + activation_loss_weight * activation_loss.mean()
```

## Weight Mapping

The implementation handles weight mapping between HuggingFace and MLX formats:

- **Loading**: `sanitize()` converts HF → MLX format
- **Saving**: `unsanitize()` converts MLX → HF format

## Requirements

- The model path should contain:
  - `config.json`
  - `tokenizer.json` or tokenizer files
  - Model weights (`.safetensors` or `.bin`)

## Calibration Datasets

The tool supports several standard datasets for calibration:

### MS MARCO (`--calibration-dataset msmarco`)

- **Purpose**: Optimized for retrieval and search tasks
- **Content**: Passages from web documents
- **Best for**: Search engines, document retrieval systems

### STS-B (`--calibration-dataset stsb`)

- **Purpose**: Optimized for semantic similarity tasks
- **Content**: Sentence pairs with similarity scores
- **Best for**: Semantic search, duplicate detection

### Natural Questions (`--calibration-dataset nq`)

- **Purpose**: Optimized for question-answering tasks
- **Content**: Real Google search queries
- **Best for**: QA systems, conversational AI

### Mixed (`--calibration-dataset mixed`) - Default

- **Purpose**: General-purpose quantization
- **Content**: 25% questions, 25% passages, 25% sentences, 25% definitions
- **Best for**: Multi-purpose embedding models

### Custom (`--calibration-dataset custom`)

- **Purpose**: Use your own dataset
- **Content**: Specified by `--data-path`
- **Best for**: Domain-specific applications

## Tips

1. **Memory**: Use smaller batch sizes for larger models
2. **Convergence**: Monitor loss curves and adjust learning rate
3. **Quality**: Higher activation loss weight preserves more detail
4. **Speed**: Reduce activation layer step for faster training
5. **Dataset Selection**: Choose calibration data that matches your use case
6. **Sample Size**: 2048 samples typically sufficient; use more for specialized domains

## Evaluation

After quantization, evaluate your model using standard benchmarks:

```bash
# Evaluate on STS-B (quick test)
python evaluate_t5.py --model ./t5_dwq_quantized --task stsb

# Evaluate on MS MARCO (retrieval)
python evaluate_t5.py --model ./t5_dwq_quantized --task msmarco

# Full MTEB evaluation (comprehensive)
python evaluate_t5.py --model ./t5_dwq_quantized --mteb-tasks all
```

Key metrics to monitor:

- **STS-B Spearman**: Should be within 2-3% of original
- **MS MARCO MRR@10**: Should maintain >90% of original performance
- **Embedding Similarity**: Should be >0.95 vs original model