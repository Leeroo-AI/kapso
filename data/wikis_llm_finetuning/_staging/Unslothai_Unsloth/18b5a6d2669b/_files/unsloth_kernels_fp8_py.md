# File: `unsloth/kernels/fp8.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 615 |
| Classes | `FP8BlockQuantLinear`, `FbgemmFp8Linear_matmul`, `FP8_fbgemm_block_linear` |
| Functions | `weight_dequant_kernel`, `weight_dequant_block`, `weight_dequant`, `act_quant_kernel`, `act_quant`, `w8a8_block_fp8_matmul_triton`, `torchao_block_matmul`, `fp8_torch_block_quant_forward`, `... +5 more` |
| Imports | math, os, torch, triton, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** FP8 quantization/dequantization kernels with multiple backend support

**Mechanism:** Implements FP8 linear operations via three backends: 1) fbgemm_gpu for fastest rowwise/blockwise matmul (f8f8bf16_rowwise/blockwise), 2) torchao for blockwise FP8 GEMM, 3) Triton fallback (w8a8_block_fp8_matmul_triton). Includes Triton kernels for weight dequantization (weight_dequant) and activation quantization (act_quant). Autograd functions (FP8BlockQuantLinear, FbgemmFp8Linear_matmul) handle forward with FP8 matmul and backward with dequantized weights. Patches FbgemmFp8Linear and FP8Linear forward methods

**Significance:** Enables FP8 training/inference with automatic backend selection based on availability and GPU compatibility. Critical for memory efficiency (2x compression vs FP16) and speed on modern GPUs (H100/B100). Handles edge cases like transposed weights, non-divisible-by-8 shapes, and numerical stability issues with older FBGEMM versions
