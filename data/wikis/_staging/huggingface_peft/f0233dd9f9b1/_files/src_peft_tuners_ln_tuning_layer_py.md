# src/peft/tuners/ln_tuning/layer.py

## Overview
Implementation of the LNTuningLayer, which wraps normalization layers to enable selective fine-tuning. This layer maintains copies of the normalization layer for different adapters and switches between them.

## Class: LNTuningLayer

Inherits from both `nn.Module` and `BaseTunerLayer`. This wrapper enables adapter-based tuning of normalization layers.

### Class Attributes

- **adapter_layer_names**: `("ln_tuning_layers",)` - Tuple specifying which attributes contain adapter layers

### Initialization

**Parameters:**
- **base_layer**: The original normalization layer (LayerNorm, RMSNorm, etc.)
- **adapter_name**: Name of the first adapter

**Initialization Process:**
1. Initialize parent classes
2. Store base_layer (original normalization layer)
3. Create empty ModuleDict for ln_tuning_layers (adapter copies)
4. Call `update_layer()` to create first adapter
5. Set active adapter
6. Initialize empty merged_adapters list
7. Extract and store in_features and out_features

**Note:** in_features and out_features are extracted using `_get_in_out_features()` utility for consistency with other PEFT layers.

---

### Core Methods

#### update_layer()
Creates or updates an adapter by making a deep copy of the layer.

**Parameters:**
- **layer**: The normalization layer to copy
- **adapter_name**: Name for this adapter
- **inference_mode** (bool, default=False): Whether in inference mode
- **kwargs**: Additional arguments (unused)

**Process:**
1. Create a deep copy of the layer
2. Store in ln_tuning_layers dict under adapter_name
3. Set this adapter as active

**Key Design:**
Uses `deepcopy()` to create completely independent copies. This ensures:
- Each adapter has its own parameters
- Updates to one adapter don't affect others
- Can switch adapters without conflicts

#### enable_adapters()
Toggle adapters on or off.

**Parameters:**
- **enabled** (bool): True to enable, False to disable

**When Enabling:**
1. Set active adapters
2. Set `_disable_adapters = False`

**When Disabling:**
1. Unmerge if currently merged
2. Set requires_grad=False for all adapter layers
3. Set `_disable_adapters = True`

**Purpose:** Allows temporarily disabling adapters without removing them

#### merge()
Merges an adapter into the base layer by swapping positions.

**Parameters:**
- **adapter_names**: Optional list of adapter names to merge
- **safe_merge** (bool, default=False): Ignored (no numerical merging occurs)

**Process:**
1. Validate adapter_names using `check_adapters_to_merge()`
2. Check only one adapter being merged (raises ValueError otherwise)
3. If already merged: Warn and unmerge first
4. Swap base_layer with adapter's layer:
   - Adapter's copy becomes the new base_layer
   - Old base_layer moves into adapter's slot
5. Add adapter to merged_adapters list

**Key Behavior:**
- Not a true numerical merge (no weight combination)
- Just a position swap
- Makes adapter the "default" layer
- Can be reversed with unmerge()

**Limitation:** Only one adapter can be merged at a time. This is enforced with a ValueError.

#### unmerge()
Reverses the merge operation.

**Process:**
1. Check if currently merged (warn if not)
2. Pop the merged adapter name
3. Swap positions back:
   - Base layer returns to base_layer slot
   - Merged adapter returns to its adapter slot

**Effect:** Restores original configuration where base_layer is the untrained layer

#### forward()
Main forward pass that selects which layer to use.

**Parameters:**
- **x**: Input tensor
- **args, kwargs**: Additional arguments passed to underlying layer

**Logic:**

**If adapters disabled:**
1. Unmerge if currently merged
2. Use base_layer for forward

**If merged or no active adapters:**
1. Use base_layer for forward

**If adapters enabled:**
1. Validate only one active adapter (raises ValueError otherwise)
2. Get the active adapter name
3. Use that adapter's layer for forward

**Returns:** Output from selected layer

**Key Design:**
- Only one adapter can be active during forward
- Clean separation: either use base or use one adapter
- No blending or mixing of adapters

#### __repr__()
Custom string representation.

**Returns:** String like "ln_tuning.{parent_repr}"

Prepends "ln_tuning." to parent's repr for clear identification in model inspection.

---

## Design Philosophy

### Copy-Based Architecture
Rather than adding small adapter weights (like LoRA), LN Tuning:
- Creates full copies of normalization layers
- Switches between copies
- Updates only the active copy during training

**Why This Approach?**
1. **Simplicity**: No special forward logic needed
2. **Independence**: Each adapter is completely separate
3. **Correctness**: No risk of adapter interactions
4. **Debugging**: Easy to inspect each adapter's state

**Trade-offs:**
- More memory per adapter (full layer copy)
- But normalization layers are small (typically just scale and bias)
- For large models, this is negligible compared to total model size

### Swap-Based Merging
The merge operation doesn't numerically combine weights. Instead:
- Swaps the adapter into the base position
- Keeps the original base as a "backup"
- Can be reversed perfectly

**Why Not Numerical Merge?**
Normalization layers have no obvious way to "blend" parameters. The swap approach:
- Is lossless and reversible
- Maintains adapter independence
- Allows quick switching

### Single Adapter Constraint
Only one adapter can be active/merged at a time:
- **Forward**: One active adapter only
- **Merge**: One merged adapter only

**Why This Limitation?**
1. **Normalization semantics**: Unclear how to combine multiple normalizations
2. **Simplicity**: Reduces complexity significantly
3. **Efficiency**: No need for adapter blending logic

---

## Usage Patterns

### Training Multiple Adapters
```python
# Train adapter 1
model.set_adapter("adapter_1")
# ... train ...

# Train adapter 2
model.set_adapter("adapter_2")
# ... train ...
```

### Switching Between Adapters
```python
# Use adapter 1 for inference
model.set_adapter("adapter_1")
output1 = model(input)

# Switch to adapter 2
model.set_adapter("adapter_2")
output2 = model(input)
```

### Merging for Deployment
```python
# Merge adapter into base for slightly faster inference
layer.merge(["my_adapter"])

# Now forward uses merged adapter directly
output = model(input)

# Can unmerge later if needed
layer.unmerge()
```

---

## Error Handling

### Multiple Active Adapters
If attempting forward with >1 active adapters:
```
ValueError: Trying to run forward with N active adapters, but LN tuning does not allow inference with more than one adapter at a time
```

### Multiple Merge Attempts
If attempting to merge when already merged:
```
Warning: Already merged with {...}. Unmerging first.
```
Then automatically unmerges and proceeds.

### Multiple Adapter Merge
If attempting to merge >1 adapters simultaneously:
```
ValueError: Trying to merge N adapters, but LN tuning does not allow merging more than one adapter at a time
```

### Unmerge When Not Merged
If attempting to unmerge when not merged:
```
Warning: Already unmerged. Nothing to do.
```

---

## Integration with PEFT

### Parameter Tracking
The layer properly implements:
- `adapter_layer_names`: So PEFT knows which attributes contain adapters
- `merged_adapters`: So PEFT knows merge state
- `active_adapters`: Inherited from BaseTunerLayer for adapter selection

### State Dict Handling
Because adapters are full layer copies:
- State dicts naturally include all adapter parameters
- Loading/saving works transparently
- No special serialization logic needed

### Trainable Parameter Management
When adapter is active:
- Its parameters are trainable
- Base layer parameters are frozen
- Other adapter parameters are frozen

This is managed by the model class via `requires_grad_()` calls.

---

## Comparison with Other PEFT Layers

### vs. LoRA Layers
- **LoRA**: Adds small adapter matrices, blends outputs
- **LN Tuning**: Copies full layer, switches between them
- **LoRA**: Can run multiple adapters simultaneously
- **LN Tuning**: One adapter at a time

### vs. Prefix Tuning Layers
- **Prefix**: Adds trainable prefix to input
- **LN Tuning**: Replaces layer with trained copy
- **Prefix**: Affects all layers via attention
- **LN Tuning**: Only affects targeted normalization layers

### vs. Adapter Layers (bottleneck)
- **Adapter**: Inserts small bottleneck modules
- **LN Tuning**: Replaces entire targeted layers
- **Adapter**: More parameters per insertion
- **LN Tuning**: Fewer parameters (only norm layers)

---

## Reference
Paper: https://huggingface.co/papers/2312.11420
