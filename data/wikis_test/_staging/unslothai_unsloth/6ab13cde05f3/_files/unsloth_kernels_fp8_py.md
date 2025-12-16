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

**Purpose:** FP8 quantization and inference support

**Mechanism:** Provides multiple FP8 quantization and matrix multiplication implementations. Includes Triton-based block-quantized FP8 matmul, weight dequantization (row and block-wise), activation quantization, and integration with FBGEMM and TorchAO backends. Supports both inference and training with automatic fallback strategies based on GPU capabilities. Patches forward functions of FP8Linear layers for compiled model support.

**Significance:** Enables efficient FP8 inference and training on supported hardware, reducing memory requirements by ~50% compared to FP16. Provides multiple backend options with automatic selection based on GPU type and version. Critical for deploying large models on memory-constrained devices while maintaining numerical stability.
