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

**Purpose:** Validates perplexity metrics for Llama 3.1 8B models after training and merging to ensure the larger 8B parameter variant maintains quality through Unsloth's optimization pipeline.

**Mechanism:** Fine-tunes Llama 3.1 8B using QLoRA, merges adapters into base model, computes perplexity on evaluation datasets through batched inference, and generates comparison reports tracking model performance before and after fine-tuning.

**Significance:** Provides quality validation specifically for the 8B parameter Llama 3.1 variant, ensuring Unsloth's optimizations scale correctly to larger models while maintaining numerical accuracy and model capabilities.
