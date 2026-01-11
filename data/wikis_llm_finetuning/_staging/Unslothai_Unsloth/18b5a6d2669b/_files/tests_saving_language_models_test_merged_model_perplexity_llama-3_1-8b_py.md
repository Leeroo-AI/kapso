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

**Purpose:** LLaMA 3.1 8B perplexity test that benchmarks merge quality for the larger 8B parameter LLaMA variant.

**Mechanism:** Trains LLaMA 3.1-8B with rank-16 LoRA for 200 steps using llama-3.1 chat template and train_on_responses_only masking. Measures perplexity at 5 stages: base 4-bit, LoRA, merged 4-bit, merged 8-bit (subprocess), merged 16-bit. Uses longer training compared to LLaMA 3.2 test (200 vs 10 steps).

**Significance:** Tests Unsloth's scalability to larger models (8B vs 3B parameters) and validates merge correctness with extended training. The longer training duration stresses numerical stability during accumulation of gradients and subsequent merge operations, catching potential precision issues that shorter tests might miss.
