# File: `tests/saving/language_models/test_merged_model_perplexity_qwen_2.5.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 311 |
| Functions | `formatting_prompts_func`, `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates merged Qwen-2.5 model quality using perplexity

**Mechanism:** Trains Qwen2.5-7B-Instruct with QLoRA for 200 steps using Alpaca prompt format (no chat template), saves merged model, then loads in 4-bit, 8-bit (subprocess), and 16-bit to compute perplexity across quantization levels

**Significance:** Tests Unsloth support for Alibaba's Qwen architecture, validating non-chat-template formatting works correctly for Chinese-English multilingual models

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
