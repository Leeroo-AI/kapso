# File: `src/peft/tuners/shira/model.py`

**Category:** model

| Property | Value |
|----------|-------|
| Lines | 143 |
| Classes | `ShiraModel` |
| Imports | __future__, layer, peft, torch, warnings |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Main model class managing SHiRA adapters, handling mask generation and layer replacement for sparse adaptation.

**Mechanism:**
- **ShiraModel class** extends `BaseTuner`:
  - `prefix = "shira_"`, `tuner_layer_cls = ShiraLayer`
  - `target_module_mapping`: Maps architectures to default target modules

- **Key Methods:**
  - `_create_and_replace()`: Creates SHiRA layers with generated masks
    - Generates mask by calling `shira_config.mask_fn(target_base_layer, r, **kwargs)`
    - Passes random_seed if mask_type is "random"
    - Updates existing ShiraLayer or creates new module
  - `_create_new_module()`: Factory method for creating Linear SHiRA layers
    - Validates fan_in_fan_out setting for torch.nn.Linear (warns if True, sets to False)
    - Generates mask using mask_fn
    - Only supports torch.nn.Linear base layers
    - Returns Linear instance with mask, adapter_name, r, fan_in_fan_out, init_weights

- **Mask Generation Flow:**
  1. Extract base layer (handles BaseTunerLayer wrapping)
  2. Call mask_fn with base_layer and r
  3. Mask_fn returns binary tensor of shape (out_features, in_features)
  4. Mask determines which of the out×in weights will be trainable

**Significance:** Core orchestrator for SHiRA sparse adaptation. The model's main responsibility is coordinating mask generation and applying it to create sparse adapter layers. The mask-based approach is elegant: instead of learning rank-r factorizations (LoRA), SHiRA directly learns sparse updates to the full weight matrix. This can be more expressive for certain adaptation tasks while maintaining the same parameter budget. The mask_fn interface provides flexibility for different sparsity patterns (random, structured, magnitude-based, etc.).
