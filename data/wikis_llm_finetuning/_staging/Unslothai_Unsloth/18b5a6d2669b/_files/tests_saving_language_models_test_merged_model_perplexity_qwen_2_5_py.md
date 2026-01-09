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

**Purpose:** Qwen 2.5 perplexity benchmark that validates Unsloth compatibility with Alibaba's Qwen architecture.

**Mechanism:** Trains Qwen2.5-7B-Instruct with rank-16 LoRA for 200 steps using Alpaca prompt format (not chat template). Measures perplexity at 5 stages: base 4-bit, LoRA, merged 4-bit, merged 8-bit (subprocess), merged 16-bit. Notably uses simplified Alpaca formatting without train_on_responses_only.

**Significance:** Validates Unsloth works with Qwen's unique architecture (GQA with sliding window, different tokenizer vocabulary). Testing a Chinese-centric model family ensures Unsloth isn't biased toward Western model designs. The Alpaca format tests compatibility with instruction-tuning datasets beyond conversational formats.
