# File: `tests/testing_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 305 |
| Functions | `require_non_cpu`, `require_non_xpu`, `require_torch_gpu`, `require_torch_multi_gpu`, `require_torch_multi_accelerator`, `require_bitsandbytes`, `require_auto_gptq`, `require_gptqmodel`, `... +13 more` |
| Imports | accelerate, contextlib, datasets, functools, numpy, os, peft, pytest, torch, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for utility functions and decorators

**Mechanism:** Provides test utility decorators (require_non_cpu, require_torch_gpu, require_bitsandbytes, etc.), helper functions for loading datasets and images, caching mechanisms for hub models, and state dict comparison utilities

**Significance:** Test coverage for testing infrastructure - reusable test utilities and fixtures
