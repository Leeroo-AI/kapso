# File: `tests/saving/vision_models/test_save_merge_qwen2.5vl32B_model_ocr_benchmark.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 287 |
| Functions | `format_data`, `find_lora_base_model` |
| Imports | datasets, jiwer, os, pandas, pathlib, qwen_vl_utils, sys, tests, torch, tqdm, ... +2 more |

## Understanding

**Status:** ✅ Documented

**Purpose:** Tests Qwen 2.5 VL 32B model OCR performance after training

**Mechanism:** Evaluates WER/CER metrics on OCR benchmarks after LoRA training and merging

**Significance:** Validates large vision model quality preservation through OCR benchmarking
