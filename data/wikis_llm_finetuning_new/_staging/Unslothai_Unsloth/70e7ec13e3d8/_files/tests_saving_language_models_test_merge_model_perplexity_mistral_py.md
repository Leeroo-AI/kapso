# File: `tests/saving/language_models/test_merge_model_perplexity_mistral.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 318 |
| Functions | `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Perplexity benchmark test that validates model quality preservation through QLoRA training and merge for Mistral-7B-v0.3.

**Mechanism:** Unlike Llama tests, uses Alpaca-style prompt formatting instead of chat templates (instruction/input/response format). Loads `unsloth/mistral-7b-v0.3` in 4-bit, applies LoRA rank 16 to projection layers, trains for 200 steps on openassistant-guanaco dataset. Computes perplexity across 5 configurations: base 4-bit, QLoRA, merged 4-bit, merged 8-bit (subprocess via `load_and_compute_8bit_ppl`), and merged 16-bit. Uses EOS token appending for proper sequence termination.

**Significance:** Tests Unsloth's merge quality for Mistral architecture, which has different attention patterns and vocabulary than Llama. The Alpaca formatting validates non-chat template workflows that some users prefer for instruction-following tasks.
