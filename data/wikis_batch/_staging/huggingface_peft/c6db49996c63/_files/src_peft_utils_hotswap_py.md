# File: `src/peft/utils/hotswap.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 630 |
| Functions | `prepare_model_for_compiled_hotswap`, `hotswap_adapter_from_state_dict`, `check_hotswap_configs_compatible`, `hotswap_adapter` |
| Imports | __future__, math, operator, other, peft, peft_types, save_and_load, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Enables rapid adapter switching without full model reloading by directly swapping adapter weights in-place.

**Mechanism:** Prepares compiled models for hotswapping, validates config compatibility between adapters, and directly loads new adapter state dicts into existing adapter layers while preserving base model and other adapter weights. Supports both compiled and non-compiled models.

**Significance:** Performance optimization feature that dramatically reduces adapter switching overhead, crucial for applications requiring frequent adapter changes (e.g., multi-task inference, A/B testing) by avoiding expensive model reloading and re-compilation.
