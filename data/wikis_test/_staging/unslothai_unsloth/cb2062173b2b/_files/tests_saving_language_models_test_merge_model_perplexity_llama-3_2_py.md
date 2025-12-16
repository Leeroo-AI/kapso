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

**Purpose:** Comprehensive perplexity testing for Llama-3.2-3B across the complete QLoRA training and merging pipeline.

**Mechanism:**
- Computes perplexity at 5 stages: base 4-bit model, QLoRA model, merged model loaded in 4-bit/8-bit/16-bit
- Uses OpenAssistant-Guanaco dataset (train split for training, eval split for perplexity)
- Trains with LoRA rank 16 for 10 steps using response-only training (masks instruction prompts)
- Uses subprocess isolation for 8-bit perplexity computation to avoid memory issues
- Produces comparison table showing perplexity degradation/improvement across loading methods

**Significance:** Quality assurance test ensuring model merge preserves learned information. Validates that merged models maintain similar perplexity when loaded in different quantization formats. Critical for detecting merge bugs that could degrade model quality.
