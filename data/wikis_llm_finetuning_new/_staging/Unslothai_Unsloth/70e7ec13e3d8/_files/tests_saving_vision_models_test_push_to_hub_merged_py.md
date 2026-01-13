# File: `tests/saving/vision_models/test_push_to_hub_merged.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 273 |
| Functions | `format_data` |
| Imports | datasets, os, pathlib, sys, tests, torch, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests the end-to-end workflow of fine-tuning a vision model with LoRA, merging adapters into the base model, and pushing the merged model to Hugging Face Hub.

**Mechanism:** The test follows a structured pipeline: (1) loads the OCR dataset with 2000 training and 200 evaluation samples, (2) formats samples into multi-modal OpenAI-style messages with images, (3) loads Qwen2-VL-2B-Instruct (smaller 2B variant for faster testing) with 4-bit quantization, (4) configures LoRA with finetune_vision_layers and finetune_language_layers enabled, (5) trains using SFTTrainer with gradient checkpointing and linear learning rate scheduler for 10 steps, (6) saves the adapter model locally, (7) pushes the merged model to Hub using push_to_hub_merged, and (8) validates by downloading the model back. The test tracks upload and download success stages, cleaning up temporary directories afterward.

**Significance:** This test validates the push_to_hub_merged functionality for vision models, which is a key user workflow for sharing fine-tuned models. Using the smaller 2B model variant makes the test faster while still validating the complete upload/download roundtrip functionality essential for model distribution.
