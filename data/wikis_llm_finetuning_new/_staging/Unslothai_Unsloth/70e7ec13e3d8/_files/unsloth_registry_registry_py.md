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

**Purpose:** Core registry module that defines the data structures and base logic for tracking supported models and their quantization variants.

**Mechanism:** Defines `QuantType` enum with quantization options (BNB 4-bit, Unsloth dynamic, GGUF, BF16, none) and corresponding HuggingFace path tags. `ModelInfo` dataclass stores model metadata with a `model_path` property that constructs HuggingFace paths (`{org}/{name}`). `ModelMeta` dataclass serves as a template for bulk registration. The `register_model()` function adds entries to `MODEL_REGISTRY` dict with duplicate detection. `_register_models()` iterates over size/instruct/quantization combinations, registering "unsloth" org variants by default and optionally the original org's model.

**Significance:** Foundation of the entire registry system. Provides the extensible base classes that model-specific files inherit from. The design allows consistent path construction across different model families while accommodating each family's unique naming conventions through subclass overrides.
