# File: `tests/saving/vision_models/test_index_file_sharded_model.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 293 |
| Functions | `format_data` |
| Imports | datasets, huggingface_hub, os, pathlib, sys, tests, torch, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests sharded model index file generation

**Mechanism:** Trains Qwen2-VL-7B on OCR dataset, saves adapter, uploads merged model to HF Hub with push_to_hub_merged(), verifies model.safetensors.index.json exists in repository using HfFileSystem, and validates model can be downloaded successfully from hub.

**Significance:** Ensures large vision models are properly sharded with correct index files when pushed to Hugging Face Hub

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
