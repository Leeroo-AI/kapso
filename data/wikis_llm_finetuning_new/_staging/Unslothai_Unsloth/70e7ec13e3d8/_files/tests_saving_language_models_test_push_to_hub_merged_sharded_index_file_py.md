# File: `tests/saving/language_models/test_push_to_hub_merged_sharded_index_file.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 223 |
| Functions | `formatting_prompts_func` |
| Imports | datasets, gc, huggingface_hub, multiprocessing, os, pandas, pathlib, sys, tests, torch, ... +4 more |

## Understanding

**Status:** Explored

**Purpose:** Tests that pushing a merged model to Hugging Face Hub correctly generates the model.safetensors.index.json file for sharded models.

**Mechanism:** The test loads Llama-3.1-8B-Instruct (a larger model that requires sharding) in 4-bit, applies LoRA adapters to attention and MLP layers, fine-tunes with SFTTrainer and train_on_responses_only for 30 steps. After calling push_to_hub_merged(), it uses HfFileSystem to verify that model.safetensors.index.json exists in the uploaded repository. This index file is required for loading sharded safetensors models. The test validates three stages: (1) successful upload, (2) presence of the index file, and (3) successful model download from Hub.

**Significance:** Validates proper handling of large models that need to be sharded during upload. The safetensors.index.json file is essential for transformers to correctly load sharded model weights, and this test ensures Unsloth's Hub integration produces valid sharded uploads.
