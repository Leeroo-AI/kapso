# File: `src/peft/utils/constants.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 362 |
| Functions | `bloom_model_postprocess_past_key_value`, `starcoder_model_postprocess_past_key_value` |
| Imports | packaging, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines global constants, model-specific mappings, and architecture-specific helper functions for PEFT operations.

**Mechanism:** Provides constants for target modules across different model architectures, transformers model postprocessing functions (for Bloom and Starcoder), embedding layer names, classification head names, and configuration defaults used throughout PEFT.

**Significance:** Essential reference file that enables PEFT to automatically determine appropriate target modules for different model architectures without user specification, and provides model-specific workarounds for edge cases like key-value cache handling.
