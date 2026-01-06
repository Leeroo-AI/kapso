# File: `tests/test_low_level_api.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 658 |
| Classes | `DummyModel`, `TestLowLevelFunctional`, `TestInjectAdapterFromStateDict`, `TestPeftStateDict`, `Outer`, `MyModel`, `MyModel`, `MyModel`, `MyModel`, `MyModel` |
| Imports | copy, diffusers, peft, platform, pytest, re, testing_utils, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for low-level PEFT API

**Mechanism:** Tests inject_adapter_in_model, get_peft_model_state_dict, set_peft_model_state_dict functions for manual adapter injection and state dict manipulation without using high-level get_peft_model

**Significance:** Test coverage for low-level adapter injection APIs
