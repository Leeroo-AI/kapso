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

**Purpose:** Core registry infrastructure that defines the model metadata system and registration mechanisms.

**Mechanism:** Defines QuantType enum (BNB, UNSLOTH, GGUF, NONE, BF16) for quantization methods, ModelInfo dataclass for storing individual model metadata (org, base_name, version, size, quant_type, etc.) with automatic name construction, ModelMeta dataclass for defining model family configurations, and MODEL_REGISTRY global dictionary. Provides register_model() function to add models to the registry with uniqueness checks, _register_models() helper to bulk register models from ModelMeta configurations across all size/instruct/quant combinations, and _check_model_info() to validate models exist on Hugging Face Hub.

**Significance:** Foundation layer of the registry system that establishes the data structures and registration patterns used by all model-specific modules, enabling systematic tracking and discovery of Unsloth-supported models with their various quantization and configuration options.
