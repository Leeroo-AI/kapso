# File: `unsloth/models/loader.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1262 |
| Classes | `FastLanguageModel`, `FastModel`, `FastVisionModel`, `FastTextModel` |
| Imports | _utils, cohere, contextlib, device_type, granite, huggingface_hub, importlib, kernels, llama, loader_utils, ... +11 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main model loader module that provides high-level APIs for loading and configuring optimized language models with various quantization options, compilation settings, and model-specific patches.

**Mechanism:** Implements three primary classes: (1) FastLanguageModel - dispatches to architecture-specific implementations (Llama, Mistral, Gemma, Qwen, etc.) based on model type detection, handling 4-bit LoRA/QLoRA loading, (2) FastModel - general model loader supporting full finetuning, 8-bit, 16-bit loading, vision models, and advanced compilation options, handles model-specific environment variable configuration for edge cases (Gemma3, Granite, GPT-OSS, etc.), (3) FastVisionModel and FastTextModel - aliases for FastModel. The from_pretrained methods handle authentication, model name mapping, PEFT adapter detection, quantization config setup, and dispatch to appropriate model classes.

**Significance:** Core user-facing module that serves as the primary entry point for loading models in Unsloth. Abstracts away complexity of model loading, quantization, and optimization configuration. Handles version compatibility checks for different transformers releases, supports pre-quantized model variants, FP8 quantization, fast inference with vLLM, and provides unified interface across dozens of model architectures.
