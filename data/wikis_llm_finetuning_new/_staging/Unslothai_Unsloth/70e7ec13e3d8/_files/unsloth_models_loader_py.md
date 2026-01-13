# File: `unsloth/models/loader.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1374 |
| Classes | `FastLanguageModel`, `FastModel`, `FastVisionModel`, `FastTextModel` |
| Imports | _utils, cohere, contextlib, device_type, granite, huggingface_hub, importlib, kernels, llama, loader_utils, ... +11 more |

## Understanding

**Status:** Explored

**Purpose:** Main model loading entry point providing `FastLanguageModel.from_pretrained()` and `FastModel.from_pretrained()` methods that handle automatic model type detection, quantization, and optimization setup.

**Mechanism:** `FastLanguageModel.from_pretrained()` accepts model name and loading options (4-bit, 8-bit, 16-bit, FP8, full finetuning), detects model architecture via `AutoConfig`/`PeftConfig`, and dispatches to appropriate FastXModel implementations (FastLlamaModel, FastMistralModel, FastQwen2Model, FastGemmaModel, etc.). Handles vLLM fast_inference integration, PEFT adapter loading, quantization config setup, and model-specific environment flags (e.g., UNSLOTH_FORCE_FLOAT32 for Gemma3). `FastModel` provides generic loading with torch.compile integration for unsupported architectures. Defines model lists like `FORCE_FLOAT32` and `DISABLE_COMPILE_MODEL_NAMES` for special handling.

**Significance:** Core component - this is the primary user-facing API for loading models with Unsloth. All model loading flows through these classes, making it the central orchestrator for optimization, quantization, and model preparation.
