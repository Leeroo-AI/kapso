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

**Purpose:** Mistral perplexity benchmark that validates merge quality for the Mistral architecture across quantization levels.

**Mechanism:** Trains Mistral-7B-v0.3 with rank-16 LoRA for 200 steps using Alpaca prompt format (instead of chat template), measures perplexity at 5 stages: base 4-bit, LoRA model, merged 4-bit, merged 8-bit (subprocess), merged 16-bit. Uses custom Alpaca formatting to extract user/assistant turns from conversational data.

**Significance:** Validates Unsloth works correctly with Mistral's architecture which differs from LLaMA (sliding window attention, different normalization). The longer training (200 vs 10 steps) and Alpaca format test different code paths. Essential for multi-architecture support validation.
