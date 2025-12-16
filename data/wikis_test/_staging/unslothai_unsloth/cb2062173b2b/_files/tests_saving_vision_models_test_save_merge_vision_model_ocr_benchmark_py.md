# File: `tests/saving/vision_models/test_save_merge_vision_model_ocr_benchmark.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 287 |
| Functions | `format_data`, `find_lora_base_model` |
| Imports | datasets, jiwer, os, pandas, pathlib, qwen_vl_utils, sys, tests, torch, tqdm, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** OCR benchmark test identical to qwen2.5vl32B version but using smaller Qwen2-VL-7B model, comparing base, adapter, and merged model performance with quantization variants.

**Mechanism:** Identical workflow to test_save_merge_qwen2.5vl32B_model_ocr_benchmark.py but using Qwen2-VL-7B-Instruct (7B vs 32B parameters): evaluates base model OCR performance on French dataset (WER/CER metrics), fine-tunes with LoRA adapters (r=16, 60 steps) and evaluates adapter performance, merges to 16-bit checkpoint, then systematically tests merged model in three quantization modes (16-bit, 4-bit, 8-bit). Generates comparison table showing accuracy differences between configurations, helping determine optimal deployment strategy for 7B vision models. Includes commented-out 4-bit merged model tests for potential future benchmarking.

**Significance:** Validates merge and quantization quality for medium-sized vision models. Provides performance comparison against 32B model, helping users choose between model sizes based on accuracy/resource tradeoffs. Essential for establishing that Unsloth's merge functionality works consistently across different model scales, and that quantization strategies remain viable after merging for production OCR applications.
