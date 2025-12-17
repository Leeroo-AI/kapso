# File: `utils/important_files.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 29 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines the list of priority model architectures that should always be tested in CI.

**Mechanism:** Exports a single constant `IMPORTANT_MODELS` containing 29 model names including foundational architectures (bert, gpt2, t5, llama), vision models (vit, clip, detr), audio models (whisper, wav2vec2), and multimodal models (llava, qwen2_5_vl, gemma3n). Used by `get_test_reports.py` when `--only-in IMPORTANT_MODELS` is specified.

**Significance:** Central configuration for test prioritization ensuring critical/widely-used models are always tested even in limited CI scenarios, balancing test coverage with resource constraints and providing a curated subset for rapid validation cycles.
