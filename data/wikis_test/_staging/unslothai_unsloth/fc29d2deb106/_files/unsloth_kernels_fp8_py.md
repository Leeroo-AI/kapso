# File: `unsloth/kernels/fp8.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 599 |
| Classes | `FP8BlockQuantLinear`, `FbgemmFp8Linear_matmul`, `FP8_fbgemm_block_linear` |
| Functions | `weight_dequant_kernel`, `weight_dequant_block`, `weight_dequant`, `act_quant_kernel`, `act_quant`, `w8a8_block_fp8_matmul_triton`, `torchao_block_matmul`, `fp8_torch_block_quant_forward`, `... +5 more` |
| Imports | math, os, torch, triton, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** FP8 quantization for fast inference and training

**Mechanism:** Implements FP8 (8-bit floating point) quantization with three backends: (1) fbgemm_gpu for rowwise quantization using f8f8bf16_rowwise kernel, (2) torchao/triton for blockwise quantization with custom Triton matmul, (3) fbgemm blockwise (experimental). Includes kernels for quantizing activations (act_quant), dequantizing weights (weight_dequant), and fused matmul operations. Auto-detects GPU support and selects optimal backend. Patches FP8Linear modules for training compatibility.

**Significance:** Critical for running large models efficiently - FP8 reduces memory by 2x vs FP16 while maintaining accuracy. The blockwise quantization (128x128 blocks) provides better quality than per-tensor/per-channel methods. Essential for Unsloth's support of DeepSeek and other FP8-quantized models.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
