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

**Purpose:** Implements block-wise FP8 quantization for inference and training with multiple backend support (FBGEMM, TorchAO, Triton) for memory-efficient model execution.

**Mechanism:** Three main paths: (1) FBGEMM GPU kernels for row-wise FP8 quantization, (2) TorchAO block-wise FP8 with per-block scaling, (3) Triton fallback (_w8a8_block_fp8_matmul) implementing block-quantized matrix multiplication. Weight dequantization uses triton kernels (weight_dequant_kernel) or simple scaling. Activation quantization uses act_quant kernel dividing inputs into blocks and computing per-block scales. Patches transformer layer forward/backward to transparently use FP8.

**Significance:** Reduces model memory footprint by ~50% through 8-bit quantization while maintaining accuracy, enabling inference of larger models on memory-constrained hardware. Critical for production deployment.
