# File: `tests/saving/language_models/test_push_to_hub_merged_sharded_index_file.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 223 |
| Functions | `formatting_prompts_func` |
| Imports | datasets, gc, huggingface_hub, multiprocessing, os, pandas, pathlib, sys, tests, torch, ... +4 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests sharded model upload with index file generation

**Mechanism:** Trains Llama-3.1-8B with QLoRA for 30 steps, uploads merged model to Hub, verifies model.safetensors.index.json exists in repo using HfFileSystem, then tests downloading to validate sharding works correctly

**Significance:** Ensures large models are properly sharded when uploaded to Hub, validating index file generation for multi-file model distributions required by models over 5GB

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
