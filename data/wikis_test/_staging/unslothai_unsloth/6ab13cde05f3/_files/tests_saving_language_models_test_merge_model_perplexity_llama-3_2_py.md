# File: `tests/saving/language_models/test_merge_model_perplexity_llama-3.2.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 259 |
| Functions | `formatting_prompts_func`, `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Documented

**Purpose:** Tests Llama 3.2 model perplexity after merging

**Mechanism:** Computes perplexity on test dataset at multiple quantization levels to validate merge quality

**Significance:** Ensures merged Llama 3.2 models maintain quality across quantization levels
