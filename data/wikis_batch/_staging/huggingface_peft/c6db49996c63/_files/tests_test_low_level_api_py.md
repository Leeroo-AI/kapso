# File: `tests/test_low_level_api.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 658 |
| Classes | `DummyModel`, `TestLowLevelFunctional`, `TestInjectAdapterFromStateDict`, `TestPeftStateDict`, `Outer`, `MyModel`, `MyModel`, `MyModel`, `MyModel`, `MyModel` |
| Imports | copy, diffusers, peft, platform, pytest, re, testing_utils, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for low-level PEFT API functions for adapter injection and state dict manipulation.

**Mechanism:** Tests `inject_adapter_in_model` for directly injecting adapters into models without using get_peft_model, including injection from state_dict for determining targets automatically. Tests `get_peft_model_state_dict` and `set_peft_model_state_dict` for extracting/loading adapter weights, with edge cases like adapter names matching module names, "lora"/"weight"/"bias" in module names. Tests modules_to_save, low_cpu_mem_usage, and compiled model support.

**Significance:** Critical for validating low-level API functions used by higher-level wrappers and external libraries (transformers, diffusers). Ensures correct adapter name handling in state dicts and compatibility with various model architectures.
