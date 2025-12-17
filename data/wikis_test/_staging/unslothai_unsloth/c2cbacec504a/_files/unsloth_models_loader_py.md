# File: `unsloth/models/loader.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1264 |
| Classes | `FastLanguageModel`, `FastModel`, `FastVisionModel`, `FastTextModel` |
| Imports | _utils, cohere, contextlib, device_type, granite, huggingface_hub, importlib, kernels, llama, loader_utils, ... +11 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Universal model loader that detects model types and dispatches to appropriate architecture-specific implementations.

**Mechanism:** Implements FastLanguageModel.from_pretrained() with logic for: AutoConfig loading, PEFT detection, model type identification, quantization strategy selection (4bit/8bit/fp8), pre-quantized model mapping, environment configuration, and routing to FastModel/FastLlamaModel/architecture-specific handlers.

**Significance:** Central orchestration point that enables single API for loading 80+ model architectures with automatic optimization selection. This is the main entry point users interact with.
