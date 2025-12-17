# File: `tests/saving/language_models/test_merge_model_perplexity_mistral.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 318 |
| Functions | `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Evaluates perplexity metrics for Mistral models through the complete training, merging, and evaluation pipeline to verify model quality after Unsloth's QLoRA workflow.

**Mechanism:** Loads Mistral models, applies QLoRA training, merges LoRA adapters, computes perplexity scores on test datasets using negative log-likelihood, and generates comparison reports with pandas DataFrames for analysis.

**Significance:** Model-specific perplexity validation for Mistral architecture, ensuring Unsloth's optimizations work correctly across different model families and that fine-tuned Mistral models maintain competitive performance metrics.
