# File: `src/peft/helpers.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 251 |
| Functions | `update_forward_signature`, `update_generate_signature`, `update_signature`, `check_if_peft_model`, `rescale_adapter_scale`, `disable_input_dtype_casting` |
| Imports | contextlib, copy, functools, inspect, peft_model, torch, tuners, types |

## Understanding

**Status:** ✅ Explored

**Purpose:** Utility functions for working with PEFT models.

**Mechanism:** `update_forward_signature()` / `update_generate_signature()` copy method signatures from base model to PeftModel for better IDE support. `check_if_peft_model()` checks if path contains PEFT adapter. `rescale_adapter_scale()` is a context manager that temporarily multiplies LoRA scaling factors - useful for WiSE-FT interpolation between base and fine-tuned models. `disable_input_dtype_casting()` context manager disables PEFT's automatic input dtype casting, needed for diffusers layerwise casting compatibility.

**Significance:** Utility helpers for advanced use cases. `rescale_adapter_scale` enables model interpolation; signature updates improve developer experience.
