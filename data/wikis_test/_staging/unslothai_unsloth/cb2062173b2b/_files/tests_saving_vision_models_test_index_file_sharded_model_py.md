# File: `tests/saving/vision_models/test_index_file_sharded_model.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 293 |
| Functions | `format_data` |
| Imports | datasets, huggingface_hub, os, pathlib, sys, tests, torch, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration test validating that sharded vision models (Qwen2-VL-7B) upload correctly to Hugging Face Hub with proper index files for model sharding.

**Mechanism:** Loads French OCR dataset (2000 train, 200 eval samples), formats data with system/user/assistant messages including images, fine-tunes Qwen2-VL-7B vision model with LoRA (r=16, alpha=32) for 10 steps using UnslothVisionDataCollator, saves adapter locally, then executes three-stage validation: (1) uploads merged model to Hub using push_to_hub_merged, (2) verifies model.safetensors.index.json exists in repository using HfFileSystem to confirm proper sharding, and (3) tests downloading model back from Hub to ensure accessibility. Tracks success of all stages in success dictionary and cleans up temporary files.

**Significance:** Essential test for large vision model deployment workflow. Validates that Unsloth correctly handles model sharding when pushing to Hub, which is critical for models too large to fit in single files. The safetensors.index.json check ensures proper weight distribution across shards, enabling efficient loading and inference of large vision models.
