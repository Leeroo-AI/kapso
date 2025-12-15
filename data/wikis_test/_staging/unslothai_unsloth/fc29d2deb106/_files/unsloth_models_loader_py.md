# File: `unsloth/models/loader.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1262 |
| Classes | `FastLanguageModel`, `FastModel`, `FastVisionModel`, `FastTextModel` |
| Imports | _utils, cohere, contextlib, device_type, granite, huggingface_hub, importlib, kernels, llama, loader_utils, ... +11 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Model loading and initialization hub for language models

**Mechanism:** This file provides the main API for loading pre-trained language models with Unsloth optimizations. It implements two key classes: (1) FastLanguageModel - handles loading optimized text-only models (Llama, Mistral, Gemma, Qwen, etc.) with support for 4-bit/8-bit/16-bit quantization, LoRA/QLoRA, and automatic model name mapping to pre-quantized variants, and (2) FastModel - universal loader for any model architecture including vision-language models with additional support for 8-bit training, full finetuning, QAT (Quantization-Aware Training), and FP8 quantization. The loader performs extensive validation (checking transformers versions, detecting PEFT adapters vs base models, resolving model architectures), applies model-specific configurations (forced float32 for certain models, disabled features for others), and integrates with vLLM for fast inference when requested.

**Significance:** This is the primary user-facing API and the most critical module in Unsloth. Users call FastLanguageModel.from_pretrained() or FastModel.from_pretrained() to load models, making it the gateway to all Unsloth optimizations. The intelligent model dispatch system routes different architectures (Llama, Mistral, Gemma, etc.) to specialized implementations, while maintaining a unified API. It handles complex edge cases like mixed PEFT/base model detection, pre-quantized model mapping, AMD GPU compatibility issues, and transformers version requirements, ensuring a smooth user experience across diverse hardware and model configurations.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
