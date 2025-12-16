# File: `tests/saving/language_models/test_push_to_hub_merged_sharded_index_file.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 223 |
| Functions | `formatting_prompts_func` |
| Imports | datasets, gc, huggingface_hub, multiprocessing, os, pandas, pathlib, sys, tests, torch, ... +4 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Extended integration test that validates push_to_hub_merged creates proper model.safetensors.index.json files for sharded models when uploading to Hugging Face Hub.

**Mechanism:** Similar to test_push_to_hub_merged but uses the larger Llama-3.1-8B-Instruct model (which produces sharded weights) and adds a verification stage using HfFileSystem to explicitly check for the presence of model.safetensors.index.json in the uploaded repository. Tests three stages: upload, safetensors index verification, and download. Tracks success/failure of each stage with detailed validation reports.

**Significance:** Critical test for ensuring large model uploads are handled correctly with proper sharding metadata. The safetensors index file is essential for loading sharded models, so this test validates that the push_to_hub_merged method creates complete and valid model repositories for production use. Particularly important for models that exceed single-file size limits.
