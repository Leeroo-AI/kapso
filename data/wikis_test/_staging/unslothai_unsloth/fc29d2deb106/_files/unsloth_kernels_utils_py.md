# File: `unsloth/kernels/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1034 |
| Functions | `is_cdna`, `calculate_settings`, `QUANT_STATE`, `get_lora_parameters`, `get_lora_parameters_bias`, `fast_linear_forward`, `matmul_lora` |
| Imports | bitsandbytes, ctypes, device_type, fp8, functools, importlib, torch, triton, typing, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core utilities for quantization, device management, and LoRA parameter handling

**Mechanism:** Provides essential infrastructure: (1) Device detection and stream management for CUDA/HIP/XPU, (2) Fast dequantization (fast_dequantize) for 4-bit NF4 weights using bitsandbytes C++ kernels with optimized streaming, (3) Fast GEMV (fast_gemv) for batch size 1 inference using 4-bit matmul, (4) LoRA parameter extraction (get_lora_parameters) supporting quantized base weights and fake quantization for QAT, (5) Fused matmul_lora combining base weight matmul with LoRA adapter application. Handles FP8, Float8Tensor, and bitsandbytes quant_state formats.

**Significance:** Foundation layer for all kernel operations - every other kernel depends on these utilities. The dequantization and LoRA functions are called on every forward pass for quantized models. The device management ensures correct stream assignment for multi-GPU setups. This file enables Unsloth to work with 4-bit quantized models efficiently, which is crucial for memory-constrained training.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
