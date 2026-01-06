# File: `utils/important_files.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 29 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines a curated list of important model architectures that should always be tested in CI.

**Mechanism:** Simple Python list constant (IMPORTANT_MODELS) containing 29 model identifiers including foundational models (bert, gpt2, t5, llama), vision models (vit, clip, detr), multimodal models (llava, qwen2_5_vl), and audio models (whisper, wav2vec2). The list covers diverse modalities and popular architectures.

**Significance:** Central configuration for CI optimization that ensures critical models are always tested even when running subset tests. Used by get_test_reports.py and other CI scripts to prioritize testing of widely-used or representative models, providing a balance between comprehensive coverage and efficient resource usage.
