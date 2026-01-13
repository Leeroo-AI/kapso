# File: `tests/saving/language_models/test_merged_model_perplexity_qwen_2.5.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 311 |
| Functions | `formatting_prompts_func`, `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Tests perplexity consistency of Qwen 2.5-7B-Instruct models across different quantization levels after QLoRA fine-tuning and merging.

**Mechanism:** The test loads a Qwen2.5-7B-Instruct model in 4-bit, applies LoRA adapters targeting attention and MLP projections (q, k, v, o, gate, down, up), fine-tunes on the OpenAssistant-Guanaco dataset using SFTTrainer for 200 steps, then saves the merged model. It evaluates perplexity at multiple stages: base model (4-bit), after QLoRA training, and after reloading the merged model in 4-bit, 8-bit (via subprocess to avoid CUDA memory conflicts), and 16-bit. The 8-bit evaluation runs in a separate subprocess using multiprocessing with spawn method to handle GPU memory properly.

**Significance:** Validates that the model merging process preserves model quality by comparing perplexity scores across different quantization precisions. This ensures that Unsloth's save_pretrained_merged functionality produces models that can be reliably reloaded and perform consistently regardless of the loading precision.
