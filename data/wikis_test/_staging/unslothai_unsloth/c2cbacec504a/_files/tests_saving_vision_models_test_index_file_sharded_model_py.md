# File: `tests/saving/vision_models/test_index_file_sharded_model.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 293 |
| Functions | `format_data` |
| Imports | datasets, huggingface_hub, os, pathlib, sys, tests, torch, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests saving large vision-language models with proper weight sharding and index file generation to handle multimodal models that require splitting across multiple files.

**Mechanism:** Trains vision-language models with LoRA, merges adapters, splits model weights into multiple shards when size exceeds limits, generates index JSON mapping layers to shard files, validates index correctness, and ensures sharded models can be correctly loaded.

**Significance:** Critical for deploying large multimodal models like vision-language transformers, ensuring Unsloth can properly handle models with both vision encoders and language decoders that require sharding for distribution and storage.
