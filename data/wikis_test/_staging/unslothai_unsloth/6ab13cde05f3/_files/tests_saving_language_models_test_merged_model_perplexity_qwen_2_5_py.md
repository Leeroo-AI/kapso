# File: `tests/saving/language_models/test_merged_model_perplexity_qwen_2.5.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 311 |
| Functions | `formatting_prompts_func`, `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Documented

**Purpose:** Tests Qwen 2.5 model perplexity after training and merging

**Mechanism:** Validates model quality through perplexity evaluation at different quantization levels

**Significance:** Ensures Qwen 2.5 models maintain quality after fine-tuning and deployment
