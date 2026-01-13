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

**Purpose:** Implements FP8 (8-bit floating point) quantization and dequantization operations with both row-wise and block-wise quantization strategies for training and inference.

**Mechanism:** Provides multiple FP8 implementations: (1) weight_dequant_kernel (Triton) for block dequantization: x * scale per block. (2) act_quant_kernel (Triton) for activation quantization: computes per-row scale as max(abs(x))/448, stores quantized values and scales. (3) w8a8_block_fp8_matmul (Triton) performs block-quantized FP8 matmul with tiled computation, loading scales per block. (4) Integration with fbgemm_gpu for hardware-accelerated FP8 ops (f8f8bf16_rowwise, f8f8bf16_blockwise) with version checks (requires >= 1.4.0 for numerical stability). (5) Integration with torchao for blockwise FP8 gemm. Autograd.Function classes (FP8BlockQuantLinear, FbgemmFp8Linear_matmul, FP8_fbgemm_block_linear) wrap these for gradient computation. Preference order: fbgemm (if available and working) > torchao > triton. Patches transformers' FP8Linear and FbgemmFp8Linear forward methods.

**Significance:** Enables training and inference with FP8 models (like DeepSeek-V3). FP8 provides 2x memory savings over FP16 with minimal accuracy loss. The multi-backend support ensures compatibility across different hardware and software configurations.
