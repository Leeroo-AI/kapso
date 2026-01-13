# File: `unsloth/models/vision.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1292 |
| Classes | `FastBaseModel` |
| Functions | `unsloth_base_fast_generate` |
| Imports | _utils, contextlib, device_type, functools, gc, inspect, kernels, math, models, os, ... +10 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides the base model loading infrastructure and `FastBaseModel` class that handles vision-language models (VLMs) and integrates with vLLM for fast inference, serving as the foundation for `FastVisionModel` and `FastModel`.

**Mechanism:** Implements `FastBaseModel` with: (1) `from_pretrained` supporting both HuggingFace and vLLM backends - loads quantized models (4bit/8bit via BitsAndBytes), handles VLM-specific processors, configures dtype/device, and optionally integrates vLLM for fast inference with `load_vllm` and `convert_vllm_to_huggingface`; (2) `get_peft_model` applying LoRA with configurable vision/language/attention/MLP layer targeting via `get_peft_regex`, QAT support, and proper gradient checkpointing; (3) `unsloth_base_fast_generate` wrapping generation with inference mode, static/hybrid KV cache configuration, and proper dtype handling; (4) `for_inference` and `for_training` methods managing gradient checkpointing, tokenizer padding side, and generation flags; (5) `post_patch_model` integrating gradient checkpointing with DDP-safe non-reentrant checkpointing for distributed training. Supports VLLM_SUPPORTED_VLM models (qwen2_5_vl, gemma3, mistral3, qwen3_vl).

**Significance:** Central infrastructure component that all model classes depend on. This file bridges HuggingFace transformers with Unsloth optimizations and optional vLLM acceleration. It handles the complex interactions between quantization, LoRA, gradient checkpointing, and distributed training while providing a unified API for both vision and language models.
