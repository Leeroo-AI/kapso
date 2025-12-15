# File: `tests/saving/vision_models/test_save_merge_qwen2.5vl32B_model_ocr_benchmark.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 287 |
| Functions | `format_data`, `find_lora_base_model` |
| Imports | datasets, jiwer, os, pandas, pathlib, qwen_vl_utils, sys, tests, torch, tqdm, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks Qwen2.5-VL-32B OCR performance

**Mechanism:** Evaluates base Qwen2.5-VL-32B model on French OCR, trains with LoRA, benchmarks adapter performance, merges to 16-bit, then benchmarks merged model loaded in 16-bit, 4-bit, and 8-bit configurations. Compares Word Error Rate (WER) and Character Error Rate (CER) across all configurations using OCRModelEvaluator.

**Significance:** Comprehensive validation that large vision models maintain OCR accuracy across quantization levels after merge

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
