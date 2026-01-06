# File: `src/peft/mixed_model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 473 |
| Classes | `PeftMixedModel` |
| Imports | __future__, accelerate, config, contextlib, os, peft, peft_model, torch, transformers, tuners, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Support for mixing different compatible adapter types in a single model.

**Mechanism:** `PeftMixedModel` wraps model with `MixedModel` tuner (from `tuners.mixed`). Unlike PeftModel, allows combining different adapter types (e.g., LoRA + LoHa) on same base model. Provides `load_adapter()`, `set_adapter()` (accepting list of names), `delete_adapter()`, `merge_and_unload()`. Checks adapter compatibility via `COMPATIBLE_TUNER_TYPES`. Currently does NOT support `save_pretrained()` or model card creation. `from_pretrained()` classmethod loads from Hub. Delegates most functionality to underlying MixedModel.

**Significance:** Advanced feature for combining multiple adapter methods. Used with `get_peft_model(model, config, mixed=True)`.
