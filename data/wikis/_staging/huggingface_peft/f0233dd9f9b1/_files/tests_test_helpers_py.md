# File: `tests/test_helpers.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 506 |
| Classes | `TestCheckIsPeftModel`, `TestScalingAdapters` |
| Imports | diffusers, peft, pytest, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for helper utility functions

**Mechanism:** Tests check_if_peft_model() for identifying PEFT models (hub, local, broken), rescale_adapter_scale() for dynamically adjusting adapter scales, and disable_input_dtype_casting() functionality across various model types

**Significance:** Test coverage for PEFT helper functions and utilities
