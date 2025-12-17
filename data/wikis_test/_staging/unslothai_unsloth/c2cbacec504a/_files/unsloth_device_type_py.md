# File: `unsloth/device_type.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 98 |
| Functions | `is_hip`, `get_device_type`, `get_device_count` |
| Imports | functools, inspect, torch, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Detects and exposes device type information (CUDA/HIP/XPU) and device count with caching for runtime efficiency.

**Mechanism:** Uses functools.cache decorator to compute device type once by checking torch.cuda.is_available(), torch.xpu.is_available(), and is_hip() flag. Determines if AMD blocksize allows pre-quantized 4-bit models. Sets device-specific constants like DEVICE_TYPE_TORCH which maps HIP to CUDA for compatibility.

**Significance:** Centralizes hardware detection logic so device-specific behavior can be applied consistently across Unsloth. Critical for supporting NVIDIA, AMD, and Intel GPUs with appropriate fallbacks.
