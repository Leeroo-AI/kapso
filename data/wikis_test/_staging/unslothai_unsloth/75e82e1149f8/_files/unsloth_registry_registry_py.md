# File: `unsloth/registry/registry.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 191 |
| Classes | `QuantType`, `ModelInfo`, `ModelMeta` |
| Functions | `register_model`, `_check_model_info`, `_register_models` |
| Imports | dataclasses, enum, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core registry infrastructure defining data structures and mechanisms for registering and tracking LLM models with their quantization variants and metadata.

**Mechanism:** Defines `QuantType` enum (BNB, UNSLOTH, GGUF, NONE, BF16) with corresponding tag mappings for Hugging Face paths. `ModelInfo` dataclass represents individual model variants with org, base_name, version, size, quantization type, and multimodal flag, providing `construct_model_name()` and `model_path` property. `ModelMeta` dataclass serves as template for model families with support for size-specific quantization types (dict or list). Global `MODEL_REGISTRY` dict stores all registered models. `register_model()` adds entries with duplicate checking, `_register_models()` generates all variants from ModelMeta (including both Unsloth and original org versions), and `_check_model_info()` validates models against Hugging Face Hub.

**Significance:** Foundation of Unsloth's model management system, enabling systematic registration of hundreds of model variants while maintaining metadata consistency. The separation of ModelInfo (instances) and ModelMeta (templates) provides clean abstraction. Support for size-specific quantization types reflects real-world constraints where larger models may not support all quantization methods. The registry pattern centralizes model discovery and prevents fragmentation across the codebase.
