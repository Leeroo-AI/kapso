# File: `unsloth/kernels/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1034 |
| Functions | `is_cdna`, `calculate_settings`, `QUANT_STATE`, `get_lora_parameters`, `get_lora_parameters_bias`, `fast_linear_forward`, `matmul_lora` |
| Imports | bitsandbytes, ctypes, device_type, fp8, functools, importlib, torch, triton, typing, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Shared utilities and helper functions

**Mechanism:** Provides infrastructure for all kernels: Triton settings calculation, device abstraction (CUDA/HIP/XPU support), quantization handling (dequantization, fast GEMV), LoRA parameter extraction with QAT support, memory stream management, and multi-GPU support. Handles compatibility across PyTorch versions and multiple backend libraries (bitsandbytes, TorchAO, FBGEMM). Implements fast_linear_forward and matmul_lora for production inference.

**Significance:** Enables kernel code to be device-agnostic and version-compatible. Provides critical infrastructure for quantized model support. Handles memory management and GPU stream coordination. The matmul_lora function is essential for efficient LoRA inference in production systems.
