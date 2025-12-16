# File: `unsloth/models/vision.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1263 |
| Classes | `FastBaseModel` |
| Functions | `unsloth_base_fast_generate` |
| Imports | _utils, contextlib, device_type, functools, gc, inspect, kernels, math, os, peft, ... +9 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Vision language model base framework

**Mechanism:** Provides FastBaseModel base class for multimodal architectures. Handles vision processor integration, auto-mapping for LoRA, quantization configuration, and training setup for vision-language models.

**Significance:** Extends unsloth optimization to vision-language models (LLaVA, Qwen-VL, etc.). Provides unified interface for combining language and vision modalities with LoRA fine-tuning.
