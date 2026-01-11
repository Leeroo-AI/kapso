# File: `tests/saving/vision_models/test_save_merge_vision_model_ocr_benchmark.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 287 |
| Functions | `format_data`, `find_lora_base_model` |
| Imports | datasets, jiwer, os, pandas, pathlib, qwen_vl_utils, sys, tests, torch, tqdm, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark test for Qwen2-VL-7B vision model fine-tuning and merging on French OCR tasks. Similar to the 32B test but targets the smaller 7B model variant.

**Mechanism:** Loads Qwen2-VL-7B-Instruct model in 4-bit quantization, applies LoRA fine-tuning (r=16, alpha=32) on French OCR dataset with 2000 training samples and 60 training steps, evaluates base model, LoRA adapter, and merged models using WER/CER metrics. Tests merged 16-bit model loaded in different quantization modes (16-bit, 4-bit, 8-bit). Compares all configurations and cleans up temporary files.

**Significance:** Validates Unsloth's vision model pipeline for the more accessible 7B model size. Serves as a regression test for model saving, merging, and quantization features. Ensures that the smaller model maintains OCR quality after fine-tuning and merging operations, providing confidence for users with limited hardware resources.
