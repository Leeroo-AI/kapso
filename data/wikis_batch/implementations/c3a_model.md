# C3A Model Implementation

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/c3a/model.py`
- **Lines**: 101
- **Purpose**: C3A model wrapper and orchestration

## Overview

C3AModel manages the application of C3A (Circular Convolution Adapter) to pretrained models, handling layer-specific block size configuration and adapter injection.

## C3AModel Class

**Inheritance**: Extends `BaseTuner`

**Class Attributes**:
- `prefix`: `"c3a_"`
- `tuner_layer_cls`: `C3ALayer`
- `target_module_mapping`: `TRANSFORMERS_MODELS_TO_C3A_TARGET_MODULES_MAPPING`

## Core Methods

### 1. `_create_and_replace`

Creates C3A layer and replaces target module.

**Algorithm**:
```python
def _create_and_replace(self, c3a_config, adapter_name, target, target_name, parent, current_key, **optional_kwargs):
    1. Validate current_key is not None

    2. Find matching block_size from pattern:
       # Get all pattern keys
       pattern_keys = list(c3a_config.block_size_pattern.keys())

       # Find key matching current layer name
       target_name_key = next(
           filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys),
           current_key  # Default to current_key if no match
       )

       # Get block_size (default if no pattern match)
       block_size = c3a_config.block_size_pattern.get(
           target_name_key,
           c3a_config.block_size
       )

    3. Build kwargs:
       kwargs = {
           "block_size": block_size,
           "init_weights": c3a_config.init_weights,
       }

    4. If target is already C3ALinear:
       target.update_layer(adapter_name, block_size, c3a_config.init_weights)

    5. Else:
       new_module = _create_new_module(c3a_config, adapter_name, target, **kwargs)
       if adapter_name != active_adapter:
           new_module.requires_grad_(False)
       _replace_module(parent, target_name, new_module, target)
```

**Key Feature**: Layer-specific block size via regex matching

### 2. `_create_new_module` (static)

Factory method for creating C3A layers.

**Implementation**:
```python
@staticmethod
def _create_new_module(c3a_config, adapter_name, target, **kwargs):
    # Extract base layer
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # Validate layer type
    if isinstance(target_base_layer, torch.nn.Linear):
        new_module = C3ALinear(target, adapter_name, **kwargs)
    else:
        raise ValueError(
            f"Target module {target} is not supported. "
            "Currently, only `torch.nn.Linear` is supported."
        )

    return new_module
```

**Supported**: Only `torch.nn.Linear`

## Layer-Specific Block Sizes

### Pattern Matching

The regex matching allows flexible configuration:

```python
config = C3AConfig(
    block_size=256,  # Default
    block_size_pattern={
        "q_proj": 256,
        "k_proj": 256,
        ".*mlp.*": 512,  # All MLP layers use 512
        "model.layers.0.*": 128  # First layer uses 128
    }
)
```

### Matching Logic

```python
# For layer "model.layers.5.self_attn.q_proj"
pattern_keys = ["q_proj", "k_proj", ".*mlp.*"]

# Match against patterns
# re.match(r".*\.q_proj$", "model.layers.5.self_attn.q_proj") -> True

# Use block_size from pattern
block_size = block_size_pattern["q_proj"]  # 256
```

## Application Workflow

```
1. User creates C3AConfig with block_size and optional block_size_pattern

2. get_peft_model(base_model, c3a_config)

3. C3AModel.__init__() → inject_adapter()

4. For each target module:
   └─> _create_and_replace()
       ├─> Match layer name against block_size_pattern
       ├─> Determine layer-specific block_size
       ├─> Create C3ALinear with that block_size
       └─> Replace module

5. Model ready (each layer may have different block_size)
```

## Example Usage

### Basic Application
```python
from transformers import AutoModelForCausalLM
from peft import C3AConfig, get_peft_model

base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

config = C3AConfig(
    block_size=256,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(base_model, config)
```

### Heterogeneous Block Sizes
```python
config = C3AConfig(
    block_size=256,  # Default for most layers
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    block_size_pattern={
        ".*attn.*": 256,      # Attention: 256
        ".*mlp.gate.*": 512,  # MLP gate: 512
        ".*mlp.up.*": 512,    # MLP up: 512
        ".*mlp.down.*": 512   # MLP down: 512
    }
)

model = get_peft_model(base_model, config)
```

### Multi-Adapter
```python
# First adapter
model = get_peft_model(base_model, config1)

# Second adapter with different block sizes
config2 = C3AConfig(
    block_size=128,
    target_modules=["q_proj", "v_proj"]
)
model.add_adapter("task2", config2)
```

## Design Patterns

### 1. Regex-Based Configuration
Flexible layer-specific settings via pattern matching

### 2. Default Fallback
If no pattern matches, use default block_size

### 3. Static Factory
`_create_new_module` is static, no shared state needed

### 4. Simple Architecture
No pre-hooks or shared parameters (unlike VeRA/VBLoRA)

## Comparison with Other PEFT Methods

### C3A vs LoRA vs VeRA vs VBLoRA

**Architecture Complexity**:
- **C3A**: Simplest (no shared state)
- **LoRA**: Simple (independent adapters)
- **VeRA**: Medium (shared matrices)
- **VBLoRA**: Complex (shared vector bank)

**Configuration Flexibility**:
- **C3A**: Layer-specific block sizes via patterns
- **LoRA**: Layer-specific ranks via rank_pattern
- **VeRA**: Single rank for all layers
- **VBLoRA**: Single config for all layers

**Parameter Efficiency**:
- **C3A**: Highest (with large block_size)
- **VeRA**: High (shared projections)
- **VBLoRA**: Medium-High (with save_only_topk_weights)
- **LoRA**: Medium

## Limitations

1. **Layer Type**: Only nn.Linear supported
2. **No Quantization**: 8-bit/4-bit not supported
3. **Float32 Requirement**: FFT needs float32 (memory overhead)
4. **Block Size Constraint**: Must divide layer dimensions

## Implementation Notes

1. **No Pre-Hook**: No shared state initialization needed
2. **Regex Matching**: Flexible but requires correct pattern syntax
3. **Single Pass**: Layer configuration determined during injection
4. **Independent Kernels**: Each layer has own circulant kernel

## Future Enhancements

Potential additions:

1. **Quantization Support**:
   ```python
   if loaded_in_8bit:
       from .bnb import Linear8bitLt
       return Linear8bitLt(...)
   ```

2. **Mixed Precision FFT**:
   - Float16 FFT where supported
   - Automatic dtype selection

3. **Dynamic Block Sizing**:
   - Auto-detect optimal block_size
   - GCD-based selection

4. **Conv Layer Support**:
   - Extend to Conv2D
   - Spatial circulant convolutions

## References

- **Paper**: https://huggingface.co/papers/2407.19342
- **Key Innovation**: FFT-based circulant matrices for parameter efficiency
- **Simple Design**: No shared state, straightforward architecture
