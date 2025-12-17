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

**Purpose:** Measures and validates perplexity scores for Llama 3.2 models after QLoRA training and merging to ensure model quality is maintained through the fine-tuning and merge process.

**Mechanism:** Trains Llama 3.2 with QLoRA, merges adapters, computes perplexity on validation datasets by calculating cross-entropy loss across sequences, and compares results between base model and merged model using parallel processing for efficiency.

**Significance:** Quality assurance test that quantitatively validates Unsloth's training and merging preserves or improves model performance for Llama 3.2, providing objective metrics for regression detection and model quality verification.
