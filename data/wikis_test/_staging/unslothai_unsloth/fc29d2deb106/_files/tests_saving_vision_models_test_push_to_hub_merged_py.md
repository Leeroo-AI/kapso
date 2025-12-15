# File: `tests/saving/vision_models/test_push_to_hub_merged.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 273 |
| Functions | `format_data` |
| Imports | datasets, os, pathlib, sys, tests, torch, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests vision model Hub upload

**Mechanism:** Trains Qwen2-VL-2B on French OCR dataset with LoRA, saves adapter locally, uploads merged model to Hugging Face Hub using push_to_hub_merged(), and validates successful download by loading the model back from the hub repository.

**Significance:** Validates complete workflow of training vision models and publishing them to Hugging Face Hub for distribution

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
