# File: `tests/saving/vision_models/test_push_to_hub_merged.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 273 |
| Functions | `format_data` |
| Imports | datasets, os, pathlib, sys, tests, torch, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests uploading vision-language models to HuggingFace Hub after training and merging to ensure multimodal models can be shared with proper vision and language components.

**Mechanism:** Trains vision-language models with LoRA on multimodal datasets, merges adapters, pushes complete model including vision encoder, language decoder, and image processor configurations to Hub, and verifies successful upload and model accessibility.

**Significance:** Validates the complete deployment pipeline for vision-language models, ensuring Unsloth-trained multimodal models with both image and text understanding capabilities can be distributed through HuggingFace Hub for community use.
