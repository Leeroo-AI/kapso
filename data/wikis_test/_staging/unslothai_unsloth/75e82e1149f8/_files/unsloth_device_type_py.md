# File: `unsloth/device_type.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 98 |
| Functions | `is_hip`, `get_device_type`, `get_device_count` |
| Imports | functools, inspect, torch, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Detects GPU device type and capabilities for hardware-specific optimizations.

**Mechanism:**
- `is_hip()`: Checks if running on AMD ROCm by inspecting `torch.version.hip`
- `get_device_type()`: Returns "cuda", "hip", or "xpu" based on available accelerators
- `get_device_count()`: Returns number of available GPUs
- Sets module-level constants: `DEVICE_TYPE`, `DEVICE_TYPE_TORCH`, `DEVICE_COUNT`
- `DEVICE_TYPE_TORCH` maps "hip" to "cuda" for PyTorch autocast compatibility
- `ALLOW_PREQUANTIZED_MODELS`: False on AMD due to blocksize differences (64 vs 128)
- `ALLOW_BITSANDBYTES`: Checks bitsandbytes version compatibility on AMD

**Significance:** Core infrastructure for multi-hardware support. All other modules use these constants to adapt behavior for NVIDIA, AMD, or Intel GPUs.
