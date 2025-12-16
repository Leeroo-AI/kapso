# File: `unsloth/models/loader.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1262 |
| Classes | `FastLanguageModel`, `FastModel`, `FastVisionModel`, `FastTextModel` |
| Imports | _utils, cohere, contextlib, device_type, granite, huggingface_hub, importlib, kernels, llama, loader_utils, ... +11 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Model loading and optimization pipeline

**Mechanism:** Provides unified APIs (FastLanguageModel, FastModel, FastVisionModel) to load pretrained models with Unsloth optimizations. Handles multiple loading modes (4bit QLoRA, 8bit, 16bit, full finetuning, FP8), supports various model architectures (Llama, Mistral, Qwen2/3, Granite, etc.), manages quantization modes, and applies compiler optimizations. Converts pre-quantized model names, manages LoRA adapter loading, and patches models with performance enhancements.

**Significance:** This is the central model loading mechanism that abstracts away complexity of different quantization schemes and model types. Users call from_pretrained() to get an optimized model ready for training. Supports both LoRA and full finetuning with automatic optimization selection based on model architecture and configuration.
