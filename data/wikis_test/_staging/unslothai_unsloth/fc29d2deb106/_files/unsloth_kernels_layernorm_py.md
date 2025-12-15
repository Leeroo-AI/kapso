# File: `unsloth/kernels/layernorm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 225 |
| Classes | `Fast_Layernorm` |
| Functions | `layernorm_forward`, `layernorm_backward`, `fast_layernorm`, `test_layernorm`, `testing_suite_layernorm` |
| Imports | torch, triton, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Accelerated LayerNorm for standard transformer architectures

**Mechanism:** Custom Triton kernel implementation of LayerNorm following standard definition: y = (x - mean) / sqrt(var + eps) * weight + bias. Forward pass computes mean and variance, stores inverse variance for backward. Backward pass uses saved statistics to compute gradients efficiently following llm.c methodology. All operations performed in float32 internally following PyTorch conventions. Includes test suite for validation.

**Significance:** LayerNorm appears at every transformer layer - optimizing it yields significant speedups. While RMS LayerNorm is more common in modern LLMs, standard LayerNorm is still used in older architectures (GPT-2, BERT). This kernel provides drop-in acceleration with `fast_layernorm` function and patching utilities.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
