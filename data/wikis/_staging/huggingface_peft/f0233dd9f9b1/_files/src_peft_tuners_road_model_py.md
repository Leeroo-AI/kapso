# File: `src/peft/tuners/road/model.py`

**Category:** model

| Property | Value |
|----------|-------|
| Lines | 164 |
| Classes | `RoadModel` |
| Functions | `_adapter_names_pre_forward_hook` |
| Imports | __future__, config, contextlib, functools, layer, operator, peft, torch |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Main model class managing RoAd adapters with support for mixed adapter batches during inference.

**Mechanism:**
- **_adapter_names_pre_forward_hook() function**: Pre-forward hook that injects adapter_names into kwargs for mixed-batch inference

- **RoadModel class** extends `BaseTuner`:
  - `prefix = "road_"`, `tuner_layer_cls = RoadLayer`
  - `target_module_mapping`: Maps architectures to default target modules

- **Key Methods:**
  - `_create_and_replace()`: Creates RoAd layers with variant and group_size parameters
    - Tries to extract `get_apply_tensor_subclass` from quantization config (for torchao merging)
    - Handles both new module creation and updating existing RoadLayers
  - `_create_new_module()`: Factory using dispatcher pattern
    - Dispatchers tried in order: dispatch_bnb_8bit, dispatch_bnb_4bit, dispatch_default
    - First matching dispatcher wins
    - Supports Linear, Linear8bitLt, and Linear4bit
  - `_enable_peft_forward_hooks()`: Context manager for mixed-batch inference
    - Only active when `adapter_names` kwarg is passed
    - Validates adapters exist in at least one layer
    - Registers pre-forward hooks on all RoadLayers to inject adapter_names
    - Checks that batch size matches length of adapter_names list
    - Raises error if used during training (inference-only feature)
    - Uses "__base__" as special name for base model (no adapter)

- **Mixed Adapter Support:**
  - Allows different samples in a batch to use different adapters
  - Each sample can specify which adapter to use via adapter_names list
  - Enables efficient multi-tenant or multi-task serving

**Significance:** Core orchestrator for RoAd with advanced mixed-batch inference capability. The dispatcher pattern cleanly separates quantization concerns. The mixed adapter support is particularly valuable for serving scenarios where different requests need different adaptations (e.g., different languages, domains, or users) but should be batched together for efficiency. The hook-based design ensures adapter selection propagates correctly through all layers without modifying forward signatures.
