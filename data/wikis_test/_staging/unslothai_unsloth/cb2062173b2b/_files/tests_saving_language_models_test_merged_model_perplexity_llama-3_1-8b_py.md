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

**Purpose:** Perplexity validation for Llama-3.1-8B model with extended training and dataset inspection.

**Mechanism:**
- Loads Llama-3.1-8B-Instruct in 4-bit with llama-3.1 chat template
- Prints dataset samples for debugging/verification (line 142-143)
- Trains for 200 steps with response-only training on assistant responses
- Decodes tokenized training data for inspection (line 204)
- Computes perplexity across all stages: base, QLoRA, merged in 4/8/16-bit
- Uses multiprocessing for 8-bit evaluation

**Significance:** Comprehensive test for larger Llama model (8B vs 3B) with additional debugging capabilities. Validates that merge quality is maintained for larger models and longer training runs. Includes dataset inspection features useful for debugging training data formatting issues.
