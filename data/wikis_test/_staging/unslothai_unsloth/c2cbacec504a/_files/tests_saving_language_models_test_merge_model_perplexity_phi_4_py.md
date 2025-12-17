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

**Purpose:** Tests perplexity preservation for Microsoft Phi-4 models after QLoRA fine-tuning and adapter merging to validate Unsloth's compatibility with Phi model architecture.

**Mechanism:** Trains Phi-4 models using QLoRA with formatted datasets, merges trained adapters into base weights, computes perplexity by evaluating cross-entropy loss on validation sets, and tracks metrics with multiprocessing for performance.

**Significance:** Ensures Unsloth's training pipeline correctly handles Phi-4's unique architecture, validating that the smaller but highly capable Phi models maintain quality through the fine-tuning and merging process.
