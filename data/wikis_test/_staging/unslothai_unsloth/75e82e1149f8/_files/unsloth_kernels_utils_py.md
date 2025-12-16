# File: `unsloth/kernels/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1034 |
| Functions | `is_cdna`, `calculate_settings`, `QUANT_STATE`, `get_lora_parameters`, `get_lora_parameters_bias`, `fast_linear_forward`, `matmul_lora` |
| Imports | bitsandbytes, ctypes, device_type, fp8, functools, importlib, torch, triton, typing, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core utility functions for kernel operations, quantization handling, and device management.

**Mechanism:** Provides essential infrastructure: (1) Device detection and stream management for CUDA/HIP/XPU with multi-GPU support, (2) Triton kernel configuration via calculate_settings (determines block size and warp count based on problem size), (3) Quantization utilities for NF4/FP8 dequantization using bitsandbytes/fbgemm, supporting both old and new quant_state formats, (4) LoRA parameter extraction with QAT (Quantization-Aware Training) support via fake quantizers, (5) Optimized matrix multiplication paths: fast_gemv for seq_len=1 (inference), fast_dequantize for materialization, and matmul_lora for fused LoRA operations, (6) Global weight buffers for zero-copy inference, (7) Backward compatibility shims for different PyTorch/Triton versions.

**Significance:** This is the foundational utility module that all other kernel modules depend on. It abstracts away hardware differences (NVIDIA/AMD/Intel), handles the complexity of multiple quantization backends, and provides optimized code paths for different scenarios (training vs inference, quantized vs full precision). The LoRA parameter extraction with QAT support enables advanced training workflows. The global buffer management reduces memory allocations during inference. The comprehensive device and version compatibility handling ensures Unsloth works across diverse hardware and software stacks.
