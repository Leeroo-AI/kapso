# File: `tests/saving/language_models/test_merged_model_perplexity_llama-3.1-8b.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 263 |
| Functions | `formatting_prompts_func`, `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Documented

**Purpose:** Tests Llama 3.1 8B model perplexity after merging

**Mechanism:** Evaluates merged model quality through perplexity computation at various precision levels

**Significance:** Validates quality preservation for Llama 3.1 8B across quantization methods
