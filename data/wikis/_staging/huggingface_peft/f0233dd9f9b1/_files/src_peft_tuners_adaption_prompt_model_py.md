# File: `src/peft/tuners/adaption_prompt/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 169 |
| Classes | `AdaptionPromptModel` |
| Imports | config, layer, peft, torch, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Adaption prompt model wrapper

**Mechanism:** Wraps base model by replacing top L attention modules with AdaptedAttention wrappers. Maintains adapter state via _cached_adapters dict for swapping. Implements add_adapter(), set_adapter(), enable/disable_adapter_layers(). Freezes all parameters except adaption_prompt/adaption_gate via _mark_only_adaption_prompts_as_trainable().

**Significance:** Core model class for adaption prompt tuning. Manages multi-adapter pattern with state caching/swapping mechanism. Essential for prompt-based parameter-efficient fine-tuning.
