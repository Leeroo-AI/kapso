# File: `unsloth/device_type.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 98 |
| Functions | `is_hip`, `get_device_type`, `get_device_count` |
| Imports | functools, inspect, torch, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Device type detection and capability checking

**Mechanism:** Detects CUDA/HIP/XPU hardware using torch APIs with caching

**Significance:** Unified hardware abstraction layer
