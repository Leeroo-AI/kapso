# File: `tests/saving/language_models/test_merge_4bit_validation.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 248 |
| Functions | `formatting_prompts_func` |
| Imports | datasets, pathlib, sys, tests, torch, transformers, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** 4-bit merge correctness test that validates Unsloth correctly handles and prevents invalid merge operations when working with 4-bit base models.

**Mechanism:** Executes a 6-phase test: (1) trains LLaMA 3.1-8B with LoRA, (2) saves with forced_merged_4bit, (3) reloads the 4-bit merged model, (4) trains again with new LoRA adapters, (5) verifies that regular merge throws TypeError with helpful message, (6) confirms forced_merged_4bit succeeds. This tests the critical constraint that 4-bit base models require special handling.

**Significance:** Ensures data integrity by preventing users from accidentally creating corrupted models through improper merge operations. Validates Unsloth's error handling and enforces the architectural constraint that 4-bit quantized base weights cannot undergo standard 16-bit merging without precision loss warnings.
