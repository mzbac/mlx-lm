# DWQ T5 Encoder Quantization

This document describes how to use Dynamic Weight Quantization (DWQ) for T5 encoder models.

## Overview

DWQ T5 is an adaptation of the DWQ algorithm specifically designed for T5 encoder models. Unlike autoregressive language models, T5 encoders process sequences bidirectionally and output hidden states rather than logits.

## Key Features

- **T5 Encoder Support**: Adapted for bidirectional processing without token shifting
- **MSE Loss**: Uses Mean Squared Error on hidden states instead of KL divergence
- **Activation Matching**: Preserves intermediate layer activations during quantization
- **Custom Model Loading**: Supports local T5 models with proper weight mapping

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
| `--data-path` | `allenai/tulu-3-sft-mixture` | Training dataset path |
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

## Tips

1. **Memory**: Use smaller batch sizes for larger models
2. **Convergence**: Monitor loss curves and adjust learning rate
3. **Quality**: Higher activation loss weight preserves more detail
4. **Speed**: Reduce activation layer step for faster training