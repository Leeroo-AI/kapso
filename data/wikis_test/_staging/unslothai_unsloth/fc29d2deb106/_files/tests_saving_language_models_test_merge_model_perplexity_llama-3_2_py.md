# File: `tests/saving/language_models/test_merge_model_perplexity_llama-3.2.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 259 |
| Functions | `formatting_prompts_func`, `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates merged Llama-3.2 model quality using perplexity

**Mechanism:** Trains Llama-3.2-3B with QLoRA for 10 steps, saves merged model, then loads it in 4-bit, 8-bit (subprocess), and 16-bit modes to compute perplexity on eval dataset, comparing quality across quantization levels

**Significance:** Ensures merged models maintain quality across different loading modes, validating that merge operations preserve model capabilities and quantization works correctly for Llama-3.2 architecture

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
