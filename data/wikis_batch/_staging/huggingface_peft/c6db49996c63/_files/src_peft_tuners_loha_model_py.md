# File: `src/peft/tuners/loha/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 116 |
| Classes | `LoHaModel` |
| Imports | layer, peft, torch, typing |

## Understanding

**Status:** ✅ Documented

**Purpose:** Main model class that applies LoHa adapters to a pretrained model by wrapping target layers with LoHaLayer implementations.

**Mechanism:** Extends LycorisTuner base class with LoHa-specific layer mappings (Conv2d, Conv1d, Linear) and target module defaults (TRANSFORMERS_MODELS_TO_LOHA_TARGET_MODULES_MAPPING). The _create_and_replace method wraps base layers with appropriate LoHa variants, applying rank and alpha patterns from the config to allow per-layer customization.

**Significance:** High-level interface for applying LoHa adapters to models. Handles the complexity of identifying and replacing target modules while preserving the base model structure. Supports popular architectures out-of-the-box through predefined target module mappings for transformers models.
