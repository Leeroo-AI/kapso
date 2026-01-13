# File: `tests/saving/language_models/test_merged_model_perplexity_llama-3.1-8b.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 263 |
| Functions | `formatting_prompts_func`, `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Perplexity benchmark test that validates model quality preservation through QLoRA training and merge for Llama-3.1-8B-Instruct.

**Mechanism:** Loads `unsloth/Llama-3.1-8B-Instruct` in 4-bit with Flash Attention 2 or SDPA depending on bfloat16 support. Applies llama-3.1 chat template, trains with LoRA rank 16 for 200 steps using `train_on_responses_only` to mask user prompts. Prints dataset sample for debugging. Computes perplexity on openassistant-guanaco eval split across 5 configurations: base 4-bit, QLoRA model, merged 4-bit, merged 8-bit (in subprocess for memory isolation), and merged 16-bit.

**Significance:** Tests the larger Llama-3.1-8B model which is more commonly used in production. Validates that Unsloth's optimization and merge process scales correctly to 8B parameter models while preserving output quality across different quantization levels.
