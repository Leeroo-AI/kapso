# File: `tests/saving/language_models/test_merge_4bit_validation.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 248 |
| Functions | `formatting_prompts_func` |
| Imports | datasets, pathlib, sys, tests, torch, transformers, trl, unsloth |

## Understanding

**Status:** Explored

**Purpose:** Multi-phase test that validates proper error handling when attempting to merge 4-bit base models and verifies the `forced_merged_4bit` save method works correctly.

**Mechanism:** Executes in 6 phases: (1) Loads Llama-3.1-8B-Instruct in 4-bit with LoRA, (2) trains briefly, (3) saves with `save_method="forced_merged_4bit"`, (4) reloads the 4-bit merged model and applies new LoRA adapters for second training round, (5) attempts regular merge without save_method (expects TypeError since 4-bit base models cannot be merged to 16-bit), (6) successfully saves again with `forced_merged_4bit`. Validates error message contains expected text about using forced_merged_4bit for 4-bit base models.

**Significance:** Critical test ensuring Unsloth properly prevents incorrect merge operations that would produce corrupted models. Validates the forced_merged_4bit feature for iterative fine-tuning workflows where users need to continue training on previously quantized models.
