# File: `tests/saving/language_models/test_merge_4bit_validation.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 248 |
| Functions | `formatting_prompts_func` |
| Imports | datasets, pathlib, sys, tests, torch, transformers, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates 4-bit model merge restrictions and forced merge

**Mechanism:** Trains Llama-3.1-8B with LoRA, saves as forced_merged_4bit, reloads and trains again, then tests that regular merge raises TypeError while forced_merged_4bit succeeds, ensuring 4-bit base models cannot be merged to 16-bit without explicit flag

**Significance:** Critical validation that prevents users from incorrectly merging 4-bit base models (not just 4-bit quantized 16-bit models) to 16-bit, enforcing proper save_method usage to maintain model integrity

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
