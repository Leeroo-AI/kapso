# File: `tests/saving/vision_models/test_save_merge_qwen2.5vl32B_model_ocr_benchmark.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 287 |
| Functions | `format_data`, `find_lora_base_model` |
| Imports | datasets, jiwer, os, pandas, pathlib, qwen_vl_utils, sys, tests, torch, tqdm, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests Qwen 2.5 Vision-Language 32B model training, merging, and OCR performance benchmarking to validate Unsloth's support for large-scale multimodal document understanding.

**Mechanism:** Loads Qwen2.5-VL-32B model, applies LoRA fine-tuning on OCR datasets, merges adapters, runs OCR benchmarks using jiwer for character/word error rate metrics, generates performance reports with pandas, and validates text extraction quality from images.

**Significance:** Validates Unsloth's capability to handle very large vision-language models for document AI tasks, ensuring the training pipeline scales to 32B parameters while maintaining OCR accuracy critical for document processing applications.
