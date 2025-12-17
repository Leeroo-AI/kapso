# File: `tests/saving/vision_models/test_save_merge_vision_model_ocr_benchmark.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 287 |
| Functions | `format_data`, `find_lora_base_model` |
| Imports | datasets, jiwer, os, pandas, pathlib, qwen_vl_utils, sys, tests, torch, tqdm, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks OCR performance for vision-language models after Unsloth training and merging to quantitatively validate text extraction accuracy from images.

**Mechanism:** Fine-tunes vision-language models on OCR datasets with LoRA, merges adapters, runs comprehensive OCR benchmarks measuring character error rate (CER) and word error rate (WER), compares against baseline performance, and generates detailed accuracy reports.

**Significance:** Provides quantitative quality assurance for vision-language models in document understanding tasks, ensuring Unsloth's training preserves or improves OCR capabilities critical for production document AI applications.
