# File: `tests/test_vision_models.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 160 |
| Classes | `TestPastKV`, `TestResnet` |
| Imports | accelerate, gc, numpy, peft, pytest, safetensors, testing_utils, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for vision models

**Mechanism:** Tests PEFT adapters on vision models (ResNet) with Conv2D layers and modules_to_save, tests past_key_values handling in multimodal models like Llava, and vision-specific adapter configurations

**Significance:** Test coverage for computer vision model architectures
