# File: `tests/saving/vision_models/test_save_merge_qwen2.5vl32B_model_ocr_benchmark.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 287 |
| Functions | `format_data`, `find_lora_base_model` |
| Imports | datasets, jiwer, os, pandas, pathlib, qwen_vl_utils, sys, tests, torch, tqdm, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Qwen2.5-VL 32B OCR test

**Mechanism:** Trains Qwen2.5-VL-32B on French OCR, benchmarks WER/CER across base, adapter, and merged models

**Significance:** Validates large vision model OCR
