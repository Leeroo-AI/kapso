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

**Purpose:** Vision model loading and optimization for multimodal architectures.

**Mechanism:** Implements FastBaseModel with support for VLM detection, processor/tokenizer handling, PEFT adaptation, and automatic model type selection (Vision2Seq vs CausalLM).

**Significance:** Extends Unsloth to vision-language models enabling optimized VLM finetuning with proper image processor handling.
