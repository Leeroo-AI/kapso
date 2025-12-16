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

**Purpose:** FP8 (8-bit floating point) quantization and dequantization operations for memory-efficient inference and training.

**Mechanism:** Implements multiple backends for FP8 operations: FBGEMM (fastest, preferred for NVIDIA H100), TorchAO, and custom Triton kernels. Supports both row-wise and block-wise quantization schemes. The row-wise approach uses per-row scaling factors for simpler but coarser quantization. Block-wise quantization (128x128 blocks) provides finer granularity and better accuracy. Includes activation quantization (dynamic, per-batch), weight quantization (static, pre-computed), and fused FP8 matrix multiplication kernels. Automatically detects GPU compatibility and selects the best backend.

**Significance:** FP8 is critical for efficient inference and training of very large models. It reduces memory footprint by 2x compared to FP16/BF16 while maintaining acceptable accuracy. The block-wise approach is particularly important as it provides better precision than per-tensor quantization. This module handles the complexity of multiple backends, GPU-specific optimizations, and numerical stability issues (like preventing NaNs from high activation values). The automatic backend selection ensures optimal performance across different hardware (H100 vs RTX consumer GPUs).
