# File: `unsloth/registry/registry.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 191 |
| Classes | `QuantType`, `ModelInfo`, `ModelMeta` |
| Functions | `register_model` |
| Imports | dataclasses, enum, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core infrastructure for the model registry system that manages metadata about supported models and their quantization variants.

**Mechanism:** Defines three key components: (1) `QuantType` enum for quantization types (BNB, UNSLOTH, GGUF, NONE, BF16), (2) `ModelInfo` dataclass storing model metadata (org, base_name, version, size, multimodal status, instruct tags, quantization), with methods to construct model names and HuggingFace paths, (3) `ModelMeta` dataclass for bulk model registration metadata, and (4) `MODEL_REGISTRY` global dict that stores all registered models. The `register_model()` function adds models to the registry, `_register_models()` performs bulk registration from ModelMeta specifications, and `_check_model_info()` validates models exist on HuggingFace Hub.

**Significance:** Foundation of the model discovery and management system in Unsloth. Provides centralized registry enabling users to search and access pre-configured models with various quantization options. All model-specific registration files depend on this infrastructure. The registry pattern allows Unsloth to maintain a curated catalog of supported models across multiple organizations (Meta, Google, Qwen, Microsoft, Mistral, DeepSeek) with consistent naming conventions and metadata.
