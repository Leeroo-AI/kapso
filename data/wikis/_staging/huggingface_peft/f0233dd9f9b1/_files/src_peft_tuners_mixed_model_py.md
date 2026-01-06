# File: `src/peft/tuners/mixed/model.py`

**Category:** Mixed Adapter Framework

| Property | Value |
|----------|-------|
| Lines | 309 |
| Classes | `MixedModel` |
| Imports | __future__, peft, torch, tqdm, typing, warnings |

## Understanding

**Status:** Fully explored

**Purpose:** Implements `MixedModel` class that enables combining multiple different PEFT adapter types (LoRA, LoHa, LoKr, AdaLoRA, OFT, Shira) within a single model.

**Mechanism:**

### Supported Adapter Types:
```python
COMPATIBLE_TUNER_TYPES = (
    PeftType.LORA, PeftType.LOHA, PeftType.LOKR,
    PeftType.ADALORA, PeftType.OFT, PeftType.SHIRA
)
```

### Key Methods:

**1. `_check_new_adapter_config()`**:
- Validates that new adapters are compatible types
- Ensures configuration matches one of the supported adapter types

**2. `_create_and_replace()`**:
- Dispatcher that routes to appropriate adapter's `_create_and_replace()` based on config type
- Handles AdaLoRA, LoRA, LoHa, LoKr, OFT, and Shira
- Raises ValueError for unsupported types

**3. `_create_new_module()`**:
- Checks for quantization compatibility (GPTQ/8bit/4bit not yet supported for mixed)
- Dispatches module creation to appropriate adapter type
- Returns configured adapter module

**4. `_replace_module()`**:
- Swaps parent module's child with new adapter module
- Handles base_layer unwrapping
- Manages device placement for different quantization types
- Copies weight/bias/state from old to new module

**5. `set_adapter()`**:
- Activates specific adapter(s) by name
- Unmerges if model is currently merged
- Updates active_adapter attribute

**6. `_unload_and_optionally_merge()`**:
- Optionally merges adapter weights into base weights
- Recursively handles nested adapters (adapters on adapters)
- Restores ModulesToSaveWrapper for additional trainable modules
- Shows progress bar during operation

**7. `delete_adapter()`**:
- Removes adapter by name from all layers
- Updates active adapter list
- Cleans up auxiliary adapters

**Significance:** MixedModel is PEFT's most flexible adapter framework, allowing practitioners to combine different adapter types strategically. For example:
- Use LoRA for attention layers (proven effective)
- Use OFT for feed-forward layers (orthogonal benefits)
- Use LoHa for specific layers requiring more expressiveness

This enables research into optimal adapter type selection per layer and practical applications where different model components benefit from different adaptation strategies. The implementation carefully manages the complexity of multiple adapter types coexisting, handling device placement, merging, and state management uniformly.

## Key Features

- **Multi-Adapter Support**: Six compatible adapter types
- **Dynamic Dispatch**: Routes operations based on adapter type
- **Recursive Merging**: Handles nested adapters correctly
- **Device Management**: Proper placement for quantized models
- **Progress Tracking**: tqdm integration for long operations
- **Validation**: Ensures compatibility at adapter addition time

## Limitations

- **No Quantization**: GPTQ/8bit/4bit not yet supported in mixed mode
- **No Weighted Adapters**: `add_weighted_adapter()` raises NotImplementedError
- **Complexity**: More overhead than single adapter type

## Usage Pattern

```python
from peft import get_peft_model, LoraConfig, LoHaConfig

# Create mixed model
model = get_peft_model(base_model, lora_config, mixed=True)

# Add different adapter type
model.add_adapter("loha_adapter", loha_config)

# Switch between adapters
model.set_adapter(["lora_default", "loha_adapter"])
```
