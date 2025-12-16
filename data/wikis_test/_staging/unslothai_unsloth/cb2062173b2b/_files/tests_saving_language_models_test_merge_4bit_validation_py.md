# File: `tests/saving/language_models/test_merge_4bit_validation.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 248 |
| Functions | `formatting_prompts_func` |
| Imports | datasets, pathlib, sys, tests, torch, transformers, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates the 4-bit merge workflow and ensures proper error handling when attempting incompatible merge operations.

**Mechanism:**
- **Phase 1-2:** Loads Llama-3.1-8B in 4-bit, applies LoRA, trains for 10 steps
- **Phase 3:** Saves with "forced_merged_4bit" method, creating a 4-bit base model
- **Phase 4:** Reloads the 4-bit merged model and trains again with new LoRA adapters
- **Phase 5:** Tests that regular merge fails with TypeError (expects 16-bit base model)
- **Phase 6:** Validates that "forced_merged_4bit" succeeds on 4-bit base model

**Significance:** Critical validation test ensuring proper handling of 4-bit base models. Tests the constraint that regular merges require 16-bit or mxfp4 base models, and validates the "forced_merged_4bit" method for continued training on 4-bit checkpoints. Prevents data corruption from incompatible merge operations.
