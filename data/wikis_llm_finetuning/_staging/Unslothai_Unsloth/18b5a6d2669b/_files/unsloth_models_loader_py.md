# File: `unsloth/models/loader.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1374 |
| Classes | `FastLanguageModel`, `FastModel`, `FastVisionModel`, `FastTextModel` |
| Imports | _utils, cohere, contextlib, device_type, granite, huggingface_hub, importlib, kernels, llama, loader_utils, ... +11 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Primary model loading interface providing FastLanguageModel, FastVisionModel, and FastTextModel classes with unified API for loading quantized models, applying LoRA, and configuring optimization settings across all supported architectures.

**Mechanism:** FastLanguageModel.from_pretrained() serves as the main entry point, handling: 1) model name resolution via get_model_name() to map between full/quantized variants, 2) quantization setup (4bit/8bit via BitsAndBytesConfig, fp8 via torchao), 3) architecture detection and routing to appropriate FastModel (Llama, Gemma, Mistral, etc.), 4) tokenizer loading and patching, 5) gradient checkpointing configuration, 6) vLLM inference setup when fast_inference=True, 7) LoRA application via get_peft_model. Uses FORCE_FLOAT32, DISABLE_COMPILE_MODEL_NAMES, DISABLE_SDPA_MODEL_NAMES lists for model-specific handling. Manages transformers version requirements (SUPPORTS_GEMMA, SUPPORTS_LLAMA31, etc.).

**Significance:** The user-facing API for all Unsloth model loading. At 1374 lines, this is one of the largest and most complex modules, reflecting the challenge of supporting 15+ model families with diverse quantization options. Every fine-tuning workflow starts here. The from_pretrained() method's flexibility (70+ parameters) enables Unsloth's "one API for all models" design philosophy.
