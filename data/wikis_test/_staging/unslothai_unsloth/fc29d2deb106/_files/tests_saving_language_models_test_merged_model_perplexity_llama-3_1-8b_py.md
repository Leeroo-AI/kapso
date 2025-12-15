# File: `tests/saving/language_models/test_merged_model_perplexity_llama-3.1-8b.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 263 |
| Functions | `formatting_prompts_func`, `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates merged Llama-3.1-8B model quality using perplexity

**Mechanism:** Trains Llama-3.1-8B with QLoRA for 200 steps using llama-3.1 chat template and train_on_responses_only, saves merged model, then loads in 4-bit, 8-bit (subprocess), and 16-bit to compute perplexity and compare quality

**Significance:** Tests the larger 8B variant of Llama-3.1, ensuring merge quality scales properly with model size and validates response-only training for instruction-tuned models

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
