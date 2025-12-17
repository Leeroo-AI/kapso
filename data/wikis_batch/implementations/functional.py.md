# Implementation: functional.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/functional.py`
- **Size**: 34 lines
- **Module**: `peft.functional`
- **Description**: Functional API for PEFT integrations with external frameworks

## Overview

This module serves as the public API surface for PEFT's functional operations, providing framework-agnostic functions that can be used by packages integrating PEFT (e.g., transformers, diffusers). It exports key utilities for adapter management, state dict manipulation, and dtype handling without requiring direct instantiation of PeftModel classes.

## Exported Functions

### Adapter Management

**cast_adapter_dtype**
```python
from peft.tuners.tuners_utils import cast_adapter_dtype
```
- **Purpose**: Casts adapter weights to specified dtype
- **Use Case**: Mixed precision training, memory optimization
- **Source**: `peft.tuners.tuners_utils`

**delete_adapter**
```python
from peft.tuners.tuners_utils import delete_adapter
```
- **Purpose**: Removes adapter from model
- **Use Case**: Dynamic adapter management, memory cleanup
- **Source**: `peft.tuners.tuners_utils`

**set_adapter**
```python
from peft.tuners.tuners_utils import set_adapter
```
- **Purpose**: Switches active adapter
- **Use Case**: Multi-adapter inference, adapter routing
- **Source**: `peft.tuners.tuners_utils`

**set_requires_grad**
```python
from peft.tuners.tuners_utils import set_requires_grad
```
- **Purpose**: Controls gradient computation for adapter parameters
- **Use Case**: Freezing/unfreezing adapters during training
- **Source**: `peft.tuners.tuners_utils`

### State Dict Operations

**get_peft_model_state_dict**
```python
from peft.utils import get_peft_model_state_dict
```
- **Purpose**: Extracts adapter-only state dict
- **Use Case**: Saving adapters separately from base model
- **Source**: `peft.utils`

**set_peft_model_state_dict**
```python
from peft.utils import set_peft_model_state_dict
```
- **Purpose**: Loads adapter state dict into model
- **Use Case**: Loading pre-trained adapters
- **Source**: `peft.utils`

### Model Injection

**inject_adapter_in_model**
```python
from peft.mapping import inject_adapter_in_model
```
- **Purpose**: Injects adapter layers into base model
- **Use Case**: Low-level adapter addition without PeftModel wrapper
- **Source**: `peft.mapping`

## Design Philosophy

### Public API Guarantee

**Stability**: Functions in this module are considered **public API**
- Semantic versioning applies
- Breaking changes require major version bump
- Safe for external packages to depend on

### Integration-Friendly

**Framework Agnostic**:
- No PeftModel dependency
- Works with raw nn.Module instances
- Compatible with transformers, diffusers, custom frameworks

## Usage Examples

### Transformers Integration

```python
from peft.functional import inject_adapter_in_model, set_adapter
from transformers import AutoModel

# Inject adapter without PeftModel wrapper
model = AutoModel.from_pretrained("bert-base")
inject_adapter_in_model(config, model, adapter_name="task1")

# Switch between adapters
set_adapter(model, "task1")
# ... inference ...
set_adapter(model, "task2")
```

### Diffusers Integration

```python
from peft.functional import get_peft_model_state_dict, set_peft_model_state_dict
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Extract UNet adapter
unet_adapter_state = get_peft_model_state_dict(pipeline.unet, adapter_name="style_adapter")

# Load into different pipeline
new_pipeline = StableDiffusionPipeline.from_pretrained("...")
set_peft_model_state_dict(new_pipeline.unet, unet_adapter_state, adapter_name="style_adapter")
```

### Manual Adapter Management

```python
from peft.functional import delete_adapter, cast_adapter_dtype

# Memory management
delete_adapter(model, "unused_adapter")

# Mixed precision
cast_adapter_dtype(model, adapter_name="main", dtype=torch.bfloat16)
```

## __all__ Export

```python
__all__ = [
    "cast_adapter_dtype",
    "delete_adapter",
    "get_peft_model_state_dict",
    "inject_adapter_in_model",
    "set_adapter",
    "set_peft_model_state_dict",
    "set_requires_grad",
]
```

**Purpose**: Explicit public API declaration
- Controls `from peft.functional import *`
- Documents intended usage
- IDE autocomplete support

## Integration Benefits

### For Framework Developers

1. **No PeftModel Wrapper**: Direct model manipulation
2. **Type Flexibility**: Works with any nn.Module subclass
3. **Fine-Grained Control**: Low-level operations exposed

### For End Users

1. **Consistent API**: Same functions across frameworks
2. **Simplified Imports**: Single entry point
3. **Documentation**: All public functions listed

## Comparison to PeftModel

### PeftModel Approach
```python
from peft import get_peft_model, LoraConfig

config = LoraConfig(...)
peft_model = get_peft_model(model, config)  # Wraps model
peft_model.set_adapter("task1")
```

### Functional Approach
```python
from peft.functional import inject_adapter_in_model, set_adapter

inject_adapter_in_model(config, model, "task1")  # Modifies in-place
set_adapter(model, "task1")  # No wrapper
```

## Cross-References

- **Source Modules**: `peft.tuners.tuners_utils`, `peft.utils`, `peft.mapping`
- **Used By**: `transformers`, `diffusers`, custom integrations
- **Related**: `peft.helpers` (user-facing utilities), `peft.peft_model` (wrapper API)
