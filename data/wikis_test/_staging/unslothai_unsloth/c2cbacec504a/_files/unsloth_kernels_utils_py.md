# File: `unsloth/kernels/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1034 |
| Functions | `is_cdna`, `calculate_settings`, `QUANT_STATE`, `get_lora_parameters`, `get_lora_parameters_bias`, `fast_linear_forward`, `matmul_lora` |
| Imports | bitsandbytes, ctypes, device_type, fp8, functools, importlib, torch, triton, typing, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Central utilities module providing Triton configuration, quantization/dequantization, LoRA parameter extraction, and multi-device CUDA/XPU stream management for kernel execution.

**Mechanism:** Key utilities: (1) calculate_settings() computes BLOCK_SIZE and num_warps based on dimension (2**ceil(log2(n))), (2) fast_dequantize() handles 4-bit and FP8 weight dequantization with optional buffer reuse, (3) get_lora_parameters() extracts W, quant_state, A, B, scaling with QAT support, (4) matmul_lora() fuses base layer GEMM with LoRA update, (5) fast_gemv() optimized single-vector multiplication. Stream management via ctypes for GPU synchronization, device detection for CUDA/HIP/XPU.

**Significance:** Foundational utilities enabling all kernel optimizations - proper BLOCK_SIZE selection critical for Triton performance, dequantization efficiency essential for quantized model training, LoRA parameter management enables seamless adapter integration.
