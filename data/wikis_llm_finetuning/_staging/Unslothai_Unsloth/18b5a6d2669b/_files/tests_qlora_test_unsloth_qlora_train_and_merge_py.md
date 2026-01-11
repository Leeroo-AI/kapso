# File: `tests/qlora/test_unsloth_qlora_train_and_merge.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 211 |
| Functions | `get_unsloth_model_and_tokenizer`, `get_unsloth_peft_model` |
| Imports | datasets, itertools, pathlib, sys, tests, torch, trl, unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Unsloth QLoRA integration test that validates the complete Unsloth-accelerated QLoRA training and merging pipeline.

**Mechanism:** Uses FastLanguageModel to load LLaMA 3.2-1B in 4-bit, applies Unsloth's PEFT implementation with rank-64 LoRA on all linear layers, trains for 100 steps, then tests Unsloth's save_pretrained_merged() with merged_16bit method. Compares responses before training, after training, and after reloading the merged model to verify the entire Unsloth workflow produces correct results.

**Significance:** Primary integration test for Unsloth's core value proposition - faster QLoRA training and optimized model merging. Validates that Unsloth's optimizations don't compromise model quality or merge correctness compared to the HF baseline in test_hf_qlora_train_and_merge.py.
