# File: `unsloth/kernels/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1034 |
| Functions | `is_cdna`, `calculate_settings`, `QUANT_STATE`, `get_lora_parameters`, `get_lora_parameters_bias`, `fast_linear_forward`, `matmul_lora` |
| Imports | bitsandbytes, ctypes, device_type, fp8, functools, importlib, torch, triton, typing, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Kernel utilities providing dequantization, GEMV, matmul_lora, and device management

**Mechanism:** Core utilities: 1) fast_dequantize - NF4/FP8 weight dequantization using bitsandbytes/fbgemm_gpu, with optional global buffers for inference, 2) fast_gemv - optimized matrix-vector product for seq_len=1 using bitsandbytes 4-bit kernels, 3) matmul_lora - fused matmul with LoRA adapters (base@W + X@A@B*scale), handles 4bit/8bit/FP8/Float8Tensor, 4) get_lora_parameters - extracts weights, quant_state, LoRA adapters from proj layers, supports QAT fake quantization. Device-specific implementations for CUDA/HIP/XPU with proper stream management

**Significance:** Foundation utilities used throughout all kernel modules. Critical for quantization support (4bit/8bit/FP8), LoRA operations, and multi-backend compatibility. Handles complex edge cases like transposed weights, multiple GPU streams, global buffer reuse for inference speed. calculate_settings provides optimal Triton block size/warp count. Essential infrastructure enabling Unsloth's quantization and LoRA features
