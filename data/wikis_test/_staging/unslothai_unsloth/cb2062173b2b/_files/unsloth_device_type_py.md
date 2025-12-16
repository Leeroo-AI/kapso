# File: `unsloth/device_type.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 98 |
| Functions | `is_hip`, `get_device_type`, `get_device_count` |
| Imports | functools, inspect, torch, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Detects and configures hardware device types (NVIDIA CUDA, AMD ROCm/HIP, Intel XPU) and determines their capabilities for quantization and pre-quantized model support.

**Mechanism:**
- `is_hip()`: Cached function that detects AMD ROCm/HIP by checking `torch.version.hip`
- `get_device_type()`: Detects available accelerator in priority order (CUDA, HIP via `is_hip()`, XPU, or torch.accelerator), raises error if none found
- `get_device_count()`: Returns number of available devices based on device type
- Sets module-level constants: `DEVICE_TYPE` (cuda/hip/xpu), `DEVICE_TYPE_TORCH` (normalized for torch API calls)
- For HIP devices, checks bitsandbytes compatibility:
  - Inspects `Params4bit` source code for HIP-specific blocksize handling (64 vs 128)
  - Sets `ALLOW_PREQUANTIZED_MODELS=False` if HIP uses different blocksize (incompatible with pre-quantized models)
  - Checks bitsandbytes version > 0.48.2.dev0 for HIP support via `ALLOW_BITSANDBYTES` flag

**Significance:** This module abstracts hardware differences across GPU vendors, crucial for Unsloth's cross-platform support. The HIP blocksize detection prevents silent corruption when loading pre-quantized 4-bit models on AMD GPUs. The device type detection enables conditional code paths for vendor-specific optimizations throughout the codebase, ensuring correct behavior on NVIDIA, AMD, and Intel accelerators.
