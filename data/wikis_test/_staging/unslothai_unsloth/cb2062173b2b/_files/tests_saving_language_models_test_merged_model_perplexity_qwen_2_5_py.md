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

**Purpose:** Integration test that validates perplexity preservation across different quantization levels when fine-tuning and merging a Qwen 2.5 model with QLoRA adapters.

**Mechanism:** Loads Qwen2.5-7B-Instruct in 4-bit mode, applies LoRA adapters, trains for 200 steps on OpenAssistant dataset using Alpaca prompt format, saves the merged model, then loads the merged model in 4-bit, 8-bit (via subprocess to avoid memory conflicts), and 16-bit modes. Computes perplexity at each stage (base model, QLoRA model, and all merged variants) using the perplexity_eval utility. The 8-bit evaluation runs in a separate subprocess to properly handle memory cleanup. Results are compared using the model comparison utilities.

**Significance:** Critical quality assurance test ensuring that the save_pretrained_merged functionality maintains model quality across different quantization formats. Demonstrates that merged models can be reloaded in various bit precisions without perplexity degradation, validating the correctness of the merge operation for Qwen architecture models.
