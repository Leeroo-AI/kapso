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

**Purpose:** Implements FastBaseModel for vision-language models (VLMs) with optimized generation, vLLM integration, and specialized handling for multimodal inputs. Supports models like Llama 3.2 Vision, Qwen2-VL, and other image-text-to-text architectures.

**Mechanism:** Provides unsloth_base_fast_generate wrapper that: 1) handles diverse input formats (input_ids, input_features, input_embeds), 2) calls FastBaseModel.for_inference() to set inference mode, 3) manages logits_to_keep vs num_logits_to_keep parameters across architectures, 4) removes token_type_ids for models that don't use it, 5) applies bfloat16 mixed precision when UNSLOTH_BFLOAT16_MIXED_PRECISION=1. Defines VLLM_SUPPORTED_VLM list (qwen2_5_vl, gemma3, mistral3, qwen3_vl) and PRE_COMPILE_INFERENCE list. Uses AutoModelForVision2Seq/AutoModelForImageTextToText loaders.

**Significance:** Core infrastructure for VLM support in Unsloth. VLMs are increasingly important for multimodal applications. At 1292 lines, this is a major component handling complex generation logic, vLLM compatibility, and architecture-specific quirks. The NUM_LOGITS_TO_KEEP global dict shows the challenge of standardizing across diverse VLM architectures. Essential for modern multimodal fine-tuning workflows.
