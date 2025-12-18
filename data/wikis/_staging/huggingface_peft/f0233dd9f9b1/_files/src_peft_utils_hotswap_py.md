# File: `src/peft/utils/hotswap.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 630 |
| Functions | `prepare_model_for_compiled_hotswap`, `hotswap_adapter_from_state_dict`, `check_hotswap_configs_compatible`, `hotswap_adapter` |
| Imports | __future__, math, operator, other, peft, peft_types, save_and_load, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Enables runtime swapping of LoRA adapters without model recompilation or reloading, supporting torch.compile for efficient inference with different adapters.

**Mechanism:** prepare_model_for_compiled_hotswap() converts LoRA scalings to tensors and pads weights to target rank to prevent recompilation. hotswap_adapter() validates config compatibility, loads new weights, and swaps them in-place using torch.utils.swap_tensors or data copying for compiled models. Handles rank differences via padding/slicing.

**Significance:** Performance optimization for serving multiple LoRA adapters. Critical for production scenarios where different adapters need to be used on the same base model with minimal overhead. Works with torch.compile to avoid expensive recompilation when switching between compatible adapters of different ranks/scales.
