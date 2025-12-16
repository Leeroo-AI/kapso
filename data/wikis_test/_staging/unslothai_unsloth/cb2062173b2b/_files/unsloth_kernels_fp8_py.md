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

**Purpose:** Implements FP8 (8-bit floating point) quantization for linear layers with both row-wise and block-wise quantization schemes. Provides multiple backend implementations (FBGEMM, TorchAO, Triton) with automatic fallback, enabling efficient inference and training with reduced memory footprint.

**Mechanism:** Three quantization strategies: (1) Row-wise FP8 using FBGEMM's `f8f8bf16_rowwise` for weights with scale per row (fast for proper shapes divisible by 8); (2) Block-wise FP8 quantizing weights in blocks (128x128) for finer granularity, using FBGEMM (preferred), TorchAO (fallback #1), or custom Triton kernel (fallback #2); (3) Activation quantization using `act_quant_kernel` scaling by max value divided by 448. Implements custom autograd functions for forward/backward passes. Patches FP8Linear and FbgemmFp8Linear forward methods at module import. Detects FBGEMM capability via `test_has_fbgemm()` checking for RTX/consumer GPU compatibility issues.

**Significance:** Critical for enabling efficient finetuning and inference of large models. FP8 reduces memory usage by 2x vs FP16 while maintaining accuracy. The multi-backend approach ensures compatibility across different hardware (H100 datacenter GPUs prefer FBGEMM, consumer GPUs need Triton fallback). Block quantization provides better accuracy than row quantization for challenging cases. Automatic patching makes FP8 transparent to model code.
