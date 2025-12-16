# File: `unsloth/kernels/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1034 |
| Functions | `is_cdna`, `calculate_settings`, `QUANT_STATE`, `get_lora_parameters`, `get_lora_parameters_bias`, `fast_linear_forward`, `matmul_lora` |
| Imports | bitsandbytes, ctypes, device_type, fp8, functools, importlib, torch, triton, typing, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Central utility module providing device abstraction, quantization helpers, LoRA parameter extraction, fast dequantization, and optimized matrix operations. Handles cross-platform compatibility (CUDA/HIP/XPU) and multiple quantization backends (bitsandbytes NF4, FP8).

**Mechanism:** Key components: (1) Device abstraction - `torch_gpu_device`, `torch_device_stream`, `_get_tensor_stream` handle CUDA/XPU differences, maintain global CUDA/XPU stream arrays; (2) Triton helpers - `calculate_settings` computes optimal block sizes/warps, `triton_tanh`/`triton_cast` handle Triton 3.0+ API changes; (3) Quantization - `fast_dequantize` supports NF4 (via bitsandbytes) and FP8 with optional global buffers for inference, platform-specific implementations for CUDA/XPU; (4) LoRA - `get_lora_parameters` extracts weights/scales/adapters with QAT fake quantization support, `matmul_lora` performs fused base+adapter matmul; (5) Optimized ops - `fast_gemv` for sequence length 1, `fast_linear_forward` routes to best implementation based on context.

**Significance:** This is the foundational infrastructure module that all other kernels depend on. Provides essential abstractions for: device compatibility across NVIDIA/AMD/Intel, quantization format handling (4-bit NF4, 8-bit FP8), efficient dequantization with memory reuse, LoRA adapter integration, and optimal operation routing. The global buffer management in `fast_dequantize` is critical for inference performance. The LoRA parameter extraction supports QAT (Quantization-Aware Training) with fake quantization. Essential glue code making the entire kernel ecosystem work across platforms and quantization schemes.
