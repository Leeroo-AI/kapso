# File: `tests/saving/language_models/test_push_to_hub_merged_sharded_index_file.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 223 |
| Functions | `formatting_prompts_func` |
| Imports | datasets, gc, huggingface_hub, multiprocessing, os, pandas, pathlib, sys, tests, torch, ... +4 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests push_to_hub_merged with explicit verification of sharded model index file generation for larger language models (Llama-3.1-8B-Instruct).

**Mechanism:** Similar to test_push_to_hub_merged but uses the larger 8B model and adds a verification stage that checks for the presence of model.safetensors.index.json using HfFileSystem, ensuring proper sharding for large models before testing download functionality.

**Significance:** Validates that Unsloth correctly generates sharded model files with proper index files when pushing large merged models to Hub, which is essential for models that exceed single-file size limits.
