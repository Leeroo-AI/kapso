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

**Purpose:** Core registry infrastructure defining model metadata structures, quantization types, and model registration/lookup mechanisms.

**Mechanism:** Defines QuantType enum (BNB, UNSLOTH, GGUF, NONE, BF16) with tag mappings for HuggingFace paths; implements ModelInfo dataclass with model path property; ModelMeta dataclass for registration templates; MODEL_REGISTRY dictionary for storage; _register_models() function for creating quantized versions.

**Significance:** Provides the foundational architecture for model metadata, enabling flexible registration patterns and HuggingFace integration while supporting quantization-aware model path generation.
