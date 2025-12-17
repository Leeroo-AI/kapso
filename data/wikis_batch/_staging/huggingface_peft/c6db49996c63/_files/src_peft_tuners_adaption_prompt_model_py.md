# File: `src/peft/tuners/adaption_prompt/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 169 |
| Classes | `AdaptionPromptModel` |
| Imports | config, layer, peft, torch, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Model wrapper managing adaption prompt adapters across attention layers

**Mechanism:** AdaptionPromptModel wraps base model, stores multiple adapters in _cached_adapters dict, swaps AdaptedAttention modules in/out when switching adapters. add_adapter() wraps top L attention modules, set_adapter() enables adapter switching, enable/disable_adapter_layers() control activation. Marks only adaption_prompt/adaption_gate parameters as trainable

**Significance:** Enables multi-adapter pattern for prompt tuning - allows training/storing multiple task-specific prompt adaptations while keeping base model frozen, with efficient adapter swapping at inference time
