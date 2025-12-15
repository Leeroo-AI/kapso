# File: `tests/saving/language_models/test_merge_model_perplexity_phi_4.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 259 |
| Functions | `formatting_prompts_func`, `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates merged Phi-4 model quality using perplexity

**Mechanism:** Trains Phi-4 with QLoRA for 200 steps using phi-4 chat template and train_on_responses_only, saves merged model, then loads in 4-bit, 8-bit (subprocess), and 16-bit to compute perplexity across quantization levels

**Significance:** Tests Unsloth support for Microsoft's Phi-4 architecture, validating custom chat template handling and response-only training work correctly with smaller, efficient models

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
