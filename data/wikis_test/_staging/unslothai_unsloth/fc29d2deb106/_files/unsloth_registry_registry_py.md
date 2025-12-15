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

**Purpose:** Core registry infrastructure for model metadata

**Mechanism:** Defines QuantType enum (BNB, UNSLOTH, GGUF, NONE, BF16), ModelInfo dataclass for individual model specifications, ModelMeta dataclass for model family configurations, provides register_model() and _register_models() functions that populate the global MODEL_REGISTRY dictionary, includes _check_model_info() for HuggingFace Hub validation.

**Significance:** Foundation of Unsloth's model management system providing the data structures and registration logic used by all model-family modules. The registry pattern enables extensible model support and validation against HuggingFace Hub availability.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
