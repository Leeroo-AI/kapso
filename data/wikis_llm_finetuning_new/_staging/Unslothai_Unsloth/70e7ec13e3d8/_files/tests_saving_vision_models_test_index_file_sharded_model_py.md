# File: `tests/saving/vision_models/test_index_file_sharded_model.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 293 |
| Functions | `format_data` |
| Imports | datasets, huggingface_hub, os, pathlib, sys, tests, torch, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests vision model fine-tuning with LoRA, merged model upload to Hugging Face Hub, and validates that the sharded model index file (model.safetensors.index.json) is correctly generated.

**Mechanism:** The test pipeline: (1) loads the OCR dataset from Hugging Face (lbourdois/OCR-liboaccn-OPUS-MIT-5M-clean) with 2000 training and 200 evaluation samples, (2) formats data into OpenAI-style messages with system, user (text + image), and assistant roles, (3) loads Qwen2-VL-7B-Instruct in 4-bit quantization using FastVisionModel, (4) applies LoRA with r=16, alpha=32 targeting both vision and language layers, (5) trains using SFTTrainer with UnslothVisionDataCollator for 10 steps, (6) saves locally then pushes merged model to Hugging Face Hub, (7) verifies the presence of model.safetensors.index.json using HfFileSystem, and (8) tests downloading the uploaded model back. Success tracking monitors upload, safetensors check, and download stages.

**Significance:** This test is critical for validating that large sharded vision models are correctly saved and uploaded with proper index files. The model.safetensors.index.json file is essential for loading sharded safetensors models, and its absence would break model loading. This ensures Unsloth's vision model workflow integrates properly with the Hugging Face ecosystem.
