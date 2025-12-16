# File: `tests/saving/language_models/test_push_to_hub_merged_sharded_index_file.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 223 |
| Functions | `formatting_prompts_func` |
| Imports | datasets, gc, huggingface_hub, multiprocessing, os, pandas, pathlib, sys, tests, torch, ... +4 more |

## Understanding

**Status:** ✅ Documented

**Purpose:** Tests Hub upload with sharded model index files

**Mechanism:** Validates correct handling of large sharded models when pushing to Hub

**Significance:** Ensures large models are properly sharded and uploaded with correct index files
