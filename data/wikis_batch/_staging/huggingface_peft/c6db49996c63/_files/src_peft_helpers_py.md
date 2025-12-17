# File: `src/peft/helpers.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 251 |
| Functions | `update_forward_signature`, `update_generate_signature`, `update_signature`, `check_if_peft_model`, `rescale_adapter_scale`, `disable_input_dtype_casting` |
| Imports | contextlib, copy, functools, inspect, peft_model, torch, tuners, types |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides helper utilities for improving PEFT model usability and runtime behavior.

**Mechanism:** update_signature functions fix PeftModel method signatures to match the base model for better IDE support. check_if_peft_model validates if a path contains PEFT weights. rescale_adapter_scale temporarily adjusts LoRA scaling factors (implements wise-ft for distribution shift). disable_input_dtype_casting controls automatic dtype conversion in adapter layers.

**Significance:** Quality-of-life improvements for PEFT users. The signature updates are critical for developer experience with IDEs and documentation tools. rescale_adapter_scale provides a non-invasive way to improve inference performance when training and test distributions differ. These utilities enhance PEFT's integration with existing tools and workflows.
