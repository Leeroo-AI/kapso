# File: `unsloth/models/loader.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1262 |
| Classes | `FastLanguageModel`, `FastModel`, `FastVisionModel`, `FastTextModel` |
| Imports | _utils, cohere, contextlib, device_type, granite, huggingface_hub, importlib, kernels, llama, loader_utils, ... +11 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main entry point for loading and configuring optimized models across different architectures

**Mechanism:** Provides unified `FastLanguageModel` and `FastModel` classes that dispatch to architecture-specific implementations (Llama, Mistral, Gemma, Qwen, etc.). Handles model loading with various quantization options (4bit, 8bit, FP8, 16bit), manages device placement, processes PEFT/LoRA adapters, checks transformers version compatibility, and routes to either optimized paths for supported models or fallback compilation for unsupported ones.

**Significance:** Core orchestrator of Unsloth's model loading pipeline. This is the primary user-facing API that determines which optimizations to apply based on model type, hardware capabilities, and user configuration. It handles complex decisions like:
- Mapping model names to optimized variants (via loader_utils)
- Version checking and compatibility warnings
- Quantization configuration (BnB, FP8)
- Device-specific optimizations
- PEFT adapter loading
- Fast inference setup (vLLM integration)
- Model compilation and patching
- Special handling for vision/multimodal models via FastVisionModel
