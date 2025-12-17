# SHiRA Model Implementation

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/shira/model.py`
- **Lines**: 142
- **Purpose**: SHiRA model wrapper and orchestration logic

## Overview

This module implements `ShiraModel`, which manages the application of SHiRA (Sparse High Rank Adapter) to pretrained models. It handles mask generation, layer wrapping, and adapter injection while ensuring proper sparse pattern initialization.

## ShiraModel Class

**Inheritance**: Extends `BaseTuner` from `peft.tuners.tuners_utils`

**Class Attributes**:
- `prefix`: `"shira_"` - Parameter name prefix
- `tuner_layer_cls`: `ShiraLayer` - Base layer class
- `target_module_mapping`: `TRANSFORMERS_MODELS_TO_SHIRA_TARGET_MODULES_MAPPING` - Default target mappings

## Core Methods

### 1. `_create_and_replace(shira_config, adapter_name, target, target_name, parent, current_key, **optional_kwargs)`

**Purpose**: Creates SHiRA layer and replaces target module

**Algorithm**:
```python
def _create_and_replace(self, shira_config, adapter_name, target, target_name, parent, current_key, **optional_kwargs):
    1. Validate current_key is not None

    2. Extract bias configuration:
       bias = hasattr(target, "bias") and target.bias is not None

    3. Build kwargs dictionary:
       kwargs = {"bias": bias}
       if shira_config.mask_type == "random":
           kwargs["random_seed"] = shira_config.random_seed
       # Add any optional_kwargs
       for k, v in optional_kwargs.items():
           kwargs[k] = v

    4. If target is already SHiRA Linear:
       # Generate mask for this layer
       mask = (
           shira_config.mask_fn(target.base_layer, shira_config.r, **kwargs)
           if shira_config.mask_fn is not None
           else None
       )
       # Update existing layer
       target.update_layer(
           adapter_name,
           mask,
           shira_config.r,
           init_weights=shira_config.init_weights,
       )

    5. Else (new layer):
       # Create new SHiRA module
       new_module = _create_new_module(shira_config, adapter_name, target, **kwargs)
       if adapter_name not in active_adapter:
           new_module.requires_grad_(False)
       # Replace in model
       _replace_module(parent, target_name, new_module, target)
```

**Key Features**:
- Generates unique mask for each layer
- Passes random_seed through kwargs for reproducibility
- Supports updating existing SHiRA layers (multi-adapter)
- Handles optional kwargs for custom mask functions

### 2. `_create_new_module(shira_config, adapter_name, target, **kwargs)` (static)

**Purpose**: Factory method for creating appropriate SHiRA layer type

**Implementation**:
```python
@staticmethod
def _create_new_module(shira_config, adapter_name, target, **kwargs):
    fan_in_fan_out = shira_config.fan_in_fan_out
    _ = kwargs.pop("bias", False)  # Remove bias from kwargs

    # Extract base layer if nested
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # Validate layer type
    if isinstance(target_base_layer, torch.nn.Linear):
        if fan_in_fan_out:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            fan_in_fan_out = shira_config.fan_in_fan_out = False
    else:
        raise ValueError(
            f"Target module {target} is not supported. "
            "Currently, only the following modules are supported: `torch.nn.Linear`."
        )

    # Generate mask for this layer
    mask = (
        shira_config.mask_fn(target_base_layer, shira_config.r, **kwargs)
        if shira_config.mask_fn is not None
        else None
    )

    # Create SHiRA Linear layer
    new_module = Linear(
        target,
        mask,
        adapter_name,
        shira_config.r,
        fan_in_fan_out,
        init_weights=shira_config.init_weights,
        **kwargs,
    )

    return new_module
```

**Supported Layers**:
- `torch.nn.Linear`: Standard linear layers
- **Not Supported**: Conv1D, Conv2D, other layer types

**fan_in_fan_out Handling**:
- Warns and corrects if set to True for nn.Linear
- Currently has no effect (only Linear supported)

**Mask Generation**:
- Calls `mask_fn` for each layer independently
- Passes through all kwargs (including random_seed)
- Each layer gets unique sparse pattern

## SHiRA Application Workflow

```
1. User creates ShiraConfig
   └─> Specifies r, target_modules, mask_type, random_seed

2. get_peft_model(base_model, shira_config)
   └─> Creates ShiraModel instance

3. ShiraModel.__init__()
   └─> Calls inject_adapter()

4. inject_adapter()
   └─> For each target module:
       └─> _create_and_replace()
           ├─> Extracts bias configuration
           ├─> Builds kwargs (including random_seed)
           ├─> Calls mask_fn(base_layer, r, **kwargs)
           │   └─> Generates unique mask for this layer
           ├─> Creates Linear(target, mask, ...)
           └─> Replaces original module

5. Model ready for training/inference
```

## Mask Generation Per Layer

### Random Seed Behavior

**With random_seed set**:
```python
config = ShiraConfig(r=32, random_seed=42)

# Each layer gets same seed but different pattern
# Because mask_fn is called with same seed but different base_layer
layer1: mask_fn(layer1_base, r=32, random_seed=42)
layer2: mask_fn(layer2_base, r=32, random_seed=42)
# Patterns are reproducible but different due to layer-specific permutation
```

**Without random_seed**:
```python
config = ShiraConfig(r=32, random_seed=None)

# Each layer gets truly random pattern
layer1: mask_fn(layer1_base, r=32)  # random
layer2: mask_fn(layer2_base, r=32)  # different random
# Non-reproducible
```

### Mask Function Calling

For each layer:
```python
if shira_config.mask_fn is not None:
    mask = shira_config.mask_fn(
        target_base_layer,  # Specific to this layer
        shira_config.r,     # Same for all layers
        **kwargs            # Including random_seed if set
    )
else:
    mask = None  # Would cause error in Linear.__init__
```

## Design Patterns

### 1. Per-Layer Mask Generation
Unlike VeRA (shared matrices) or VBLoRA (shared bank):
- Each layer has independent sparse pattern
- Masks generated during layer creation
- No shared structure across layers

### 2. Factory Pattern
`_create_new_module` dispatches based on layer type:
```python
if isinstance(target_base_layer, torch.nn.Linear):
    return Linear(...)
else:
    raise ValueError("Unsupported")
```

### 3. Kwargs Pass-Through
Flexible parameter passing to mask functions:
```python
kwargs = {"bias": bias}
if shira_config.mask_type == "random":
    kwargs["random_seed"] = shira_config.random_seed
# Plus any optional_kwargs
mask = shira_config.mask_fn(base_layer, r, **kwargs)
```

### 4. Validation-Heavy
Extensive checks:
- current_key validation
- Layer type checking
- fan_in_fan_out correctness
- mask_fn availability

## Multi-Adapter Support

SHiRA supports multiple adapters:

```python
# First adapter
model = get_peft_model(base_model, config1)

# Second adapter
model.add_adapter("task2", config2)

# Each adapter has:
# - Independent masks per layer
# - Independent weights per layer
# - Independent scaling per layer
```

**Key Point**: Masks are regenerated for each adapter, even for same layer.

## Example Usage Patterns

### Basic Application
```python
from transformers import AutoModelForCausalLM
from peft import ShiraConfig, get_peft_model

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Create SHiRA config
config = ShiraConfig(
    r=32,
    target_modules=["q_proj", "v_proj"],
    mask_type="random",
    random_seed=42
)

# Apply SHiRA
model = get_peft_model(base_model, config)

# Each targeted layer now has:
# - Unique sparse mask (generated with seed=42)
# - Sparse weight parameters
# - Forward pass with sparse adaptation
```

### Custom Mask Function
```python
def my_mask_fn(base_layer, r, random_seed=None, block_size=16):
    """Block-structured sparse mask"""
    m, n = base_layer.weight.shape
    num_params = r * (m + n)

    # Custom logic...
    mask = create_block_sparse_pattern(m, n, num_params, block_size)

    return mask

config = ShiraConfig(r=32, target_modules=["q_proj", "v_proj"])
config.mask_fn = my_mask_fn
model = get_peft_model(base_model, config)
```

### Multi-Adapter with Different Patterns
```python
# Adapter 1: Random mask
config1 = ShiraConfig(
    r=32,
    target_modules=["q_proj", "v_proj"],
    mask_type="random",
    random_seed=42
)
model = get_peft_model(base_model, config1)

# Adapter 2: Custom mask
config2 = ShiraConfig(
    r=32,
    target_modules=["q_proj", "v_proj"]
)
config2.mask_fn = block_sparse_mask
model.add_adapter("task2", config2)

# Each adapter has different sparse patterns
```

## Limitations

### 1. Layer Type Support
**Only nn.Linear supported**:
```python
if isinstance(target_base_layer, torch.nn.Linear):
    # OK
else:
    raise ValueError("Target module {target} is not supported...")
```

Conv1D, Conv2D, and other layer types not supported.

### 2. Nested Base Layers
From layer.py:
```python
if self.base_layer is not self.get_base_layer():
    raise ValueError("SHiRA does not support nested base layers")
```

Cannot wrap already-adapted layers (except when adding new adapter).

### 3. Mask Persistence
Masks are generated dynamically, not stored in config:
- Must have same mask_fn when loading
- Must have same random_seed for reproducibility
- Custom mask_fn must be reapplied

### 4. No Quantization Support
Unlike VeRA and LoRA:
- No 8-bit support
- No 4-bit support
- Standard float32/float16 only

## Comparison with Other PEFT Methods

### SHiRA vs LoRA vs VeRA

**Parameter Count** (m×n layer, budget r):

| Method | Trainable Params | Shared Params | Total |
|--------|-----------------|---------------|-------|
| LoRA | r(m + n) | 0 | r(m + n) |
| VeRA | r + n | r(m + n) frozen | r + n trainable |
| SHiRA | r(m + n) | 0 | r(m + n) |

**Effective Rank**:

| Method | Max Rank | Structure |
|--------|----------|-----------|
| LoRA | r | Dense low-rank |
| VeRA | r | Dense low-rank with shared projections |
| SHiRA | min(m, n) | Sparse high-rank |

**Key Differences**:
- **LoRA**: Dense low-rank, flexible
- **VeRA**: Shared projections, very parameter-efficient
- **SHiRA**: Sparse high-rank, potentially better capacity

### When to Use SHiRA

**Use SHiRA when**:
- Need higher effective rank
- Can afford sparse operations
- Want same param count as LoRA
- Task benefits from sparse structure

**Use LoRA when**:
- Dense operations preferred
- Low-rank sufficient
- Need quantization support
- Wider tool support

**Use VeRA when**:
- Minimizing parameters critical
- Many adapters needed
- Shared structure acceptable

## Implementation Notes

1. **Mask Generation**: Called once per layer during injection
2. **Random Seed**: Passed through kwargs to mask functions
3. **No Pre-Hook**: Mask generation happens in _create_and_replace
4. **No Shared State**: Each layer independent (unlike VeRA/VBLoRA)
5. **Static Factory**: _create_new_module is static method

## Future Enhancements

Potential additions based on codebase patterns:

1. **Quantization Support**:
   ```python
   # Similar to VeRA
   if is_bnb_available() and loaded_in_8bit:
       from .bnb import Linear8bitLt
       return Linear8bitLt(...)
   ```

2. **Conv1D Support**:
   ```python
   elif isinstance(target_base_layer, Conv1D):
       # Handle fan_in_fan_out
       return Linear(...)  # with is_target_conv_1d_layer=True
   ```

3. **Dynamic Masking**:
   - Learned masks
   - Magnitude-pruning during training
   - Adaptive sparsity

4. **Structured Sparsity**:
   - Block-sparse patterns
   - N:M sparsity for hardware
   - Hierarchical patterns

## References

- **Concept**: Sparse high-rank adaptation as alternative to dense low-rank
- **Parameter Efficiency**: Same count as LoRA, higher effective rank
- **Flexibility**: Customizable sparse patterns via mask functions
- **Trade-off**: Sparse ops vs dense ops performance
