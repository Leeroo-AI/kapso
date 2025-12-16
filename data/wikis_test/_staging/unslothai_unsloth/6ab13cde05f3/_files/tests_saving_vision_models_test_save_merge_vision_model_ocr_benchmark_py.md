# File: `tests/saving/vision_models/test_save_merge_vision_model_ocr_benchmark.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 287 |
| Functions | `format_data`, `find_lora_base_model` |
| Imports | datasets, jiwer, os, pandas, pathlib, qwen_vl_utils, sys, tests, torch, tqdm, ... +2 more |

## Understanding

**Status:** ✅ Documented

**Purpose:** Tests vision model OCR accuracy after training and merging

**Mechanism:** Benchmarks OCR performance using WER/CER metrics on test datasets

**Significance:** Validates vision-language model quality for OCR tasks after fine-tuning
