# File: `tests/saving/vision_models/test_push_to_hub_merged.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 273 |
| Functions | `format_data` |
| Imports | datasets, os, pathlib, sys, tests, torch, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests push_to_hub_merged for vision-language models (Qwen2-VL) with OCR training, validating Hub upload/download for multi-modal architectures.

**Mechanism:** Loads Qwen2-VL-2B-Instruct with 4-bit quantization, applies LoRA (rank 16) to both vision and language layers using UnslothVisionDataCollator, trains for 10 steps on French OCR dataset (lbourdois/OCR-liboaccn-OPUS-MIT-5M-clean) with image-text pairs, uploads merged model via push_to_hub_merged, and validates successful download from Hub.

**Significance:** Critical test for vision-language model support, ensuring Unsloth's merge and Hub operations work correctly for multi-modal architectures that process both images and text, validating proper handling of vision encoders during the merge process.
