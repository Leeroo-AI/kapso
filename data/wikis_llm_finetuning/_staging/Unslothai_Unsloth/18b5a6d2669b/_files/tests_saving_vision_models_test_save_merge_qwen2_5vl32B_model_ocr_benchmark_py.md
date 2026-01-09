# File: `tests/saving/vision_models/test_save_merge_qwen2.5vl32B_model_ocr_benchmark.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 287 |
| Functions | `format_data`, `find_lora_base_model` |
| Imports | datasets, jiwer, os, pandas, pathlib, qwen_vl_utils, sys, tests, torch, tqdm, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive benchmark test for Qwen2.5-VL-32B model fine-tuning, merging, and quantization on French OCR tasks. Evaluates model performance across different configurations (base, LoRA adapter, 16-bit merged) and quantization levels (4-bit, 8-bit, 16-bit).

**Mechanism:** Loads the Qwen2.5-VL-32B model, fine-tunes it on French OCR dataset (2000 training samples) using LoRA with rank=16, benchmarks performance using WER/CER metrics, merges LoRA weights into base model, tests merged model across multiple quantization configurations, and compares all results. Uses OCRModelEvaluator for consistent evaluation and cleanup_utils for resource management.

**Significance:** Critical integration test that validates Unsloth's vision model training pipeline for the large 32B parameter Qwen2.5-VL model. Ensures that save/merge functionality preserves model quality across different quantization formats, which is essential for deployment scenarios where users need to balance memory constraints with performance.
