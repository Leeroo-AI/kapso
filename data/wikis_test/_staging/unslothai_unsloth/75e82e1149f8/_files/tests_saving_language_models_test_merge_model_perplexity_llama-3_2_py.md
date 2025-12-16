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

**Purpose:** Llama 3.2 perplexity testing

**Mechanism:** Trains Llama 3.2 model with LoRA, merges adapters, evaluates perplexity on base, adapter, and merged versions

**Significance:** Validates Llama 3.2 merge quality
