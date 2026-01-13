# File: `unsloth/device_type.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 127 |
| Functions | `is_hip`, `get_device_type`, `get_device_count` |
| Imports | functools, inspect, torch, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Detects and abstracts hardware device types (CUDA, HIP/ROCm, CPU) to enable cross-platform compatibility for GPU-accelerated training.

**Mechanism:** Provides `is_hip()` to detect AMD ROCm environments, `get_device_type()` to return the appropriate device string ("cuda", "hip", or "cpu"), and `get_device_count()` to enumerate available accelerators. Uses `functools.lru_cache` for efficient repeated calls. Falls back gracefully when GPU libraries are unavailable.

**Significance:** Utility - enables Unsloth to support both NVIDIA CUDA and AMD ROCm GPUs transparently, abstracting hardware differences from the rest of the codebase.
