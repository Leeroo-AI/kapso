# File: `src/peft/tuners/mixed/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 309 |
| Classes | `MixedModel` |
| Imports | __future__, peft, torch, tqdm, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Enables mixing different types of PEFT adapters (LoRA, LoHa, LoKr, AdaLoRA, OFT, Shira) within a single model, allowing different layers to use different adaptation methods.

**Mechanism:** Extends BaseTuner and delegates _create_and_replace and _create_new_module calls to the appropriate adapter-specific implementations based on the config type. Maintains compatibility by checking all adapter configs are from COMPATIBLE_TUNER_TYPES. Handles merge/unmerge recursively to support nested adapters. The _mark_only_adapters_as_trainable method freezes non-adapter parameters while respecting bias settings for each adapter type.

**Significance:** Powerful meta-adapter that enables hybrid fine-tuning strategies. Users can apply different PEFT methods to different parts of a model (e.g., LoRA for attention, LoHa for FFN), potentially achieving better performance-efficiency trade-offs. Note that weighted adapters and some quantization modes are not yet supported.
