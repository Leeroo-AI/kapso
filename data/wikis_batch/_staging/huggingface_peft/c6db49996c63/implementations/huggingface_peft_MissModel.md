# MissModel (MISS Adapter Model)

## Implementation Overview

**File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/miss/model.py`
**Lines of Code:** 130
**Language:** Python

MissModel orchestrates MISS (Minimally Structured Sparse) adapter creation and integration with pretrained models, managing adapter lifecycle from initialization to layer replacement.

## Core Implementation

### Model Class

**Class:** `MissModel(BaseTuner)`

```python
class MissModel(BaseTuner):
    """
    Creates Householder reflection adaptation (MiSS) model from a pretrained model.
    Paper: https://huggingface.co/papers/2409.15371

    Args:
        model (`torch.nn.Module`): Model to adapt
        config (`MissConfig`): Configuration
        adapter_name (`str`): Adapter name (default: "default")
        low_cpu_mem_usage (`bool`): Use meta device for initialization

    Returns:
        `torch.nn.Module`: The MiSS model
    """

    prefix: str = "miss_"
    tuner_layer_cls = MissLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_MISS_TARGET_MODULES_MAPPING
```

## Key Methods

### Layer Creation and Replacement

**Method:** `_create_and_replace()`

Creates MiSS adapters and replaces target modules:

```python
def _create_and_replace(
    self,
    miss_config,
    adapter_name,
    target,
    target_name,
    parent,
    current_key,
    **optional_kwargs,
):
    if current_key is None:
        raise ValueError("Current Key shouldn't be `None`")

    bias = hasattr(target, "bias") and target.bias is not None
    kwargs = {
        "r": miss_config.r,
        "mini_r": miss_config.mini_r,
        "miss_dropout": miss_config.miss_dropout,
        "init_weights": miss_config.init_weights,
    }
    kwargs["bias"] = bias

    # Update existing or create new MissLayer
    if not isinstance(target, MissLayer):
        new_module = self._create_new_module(miss_config, adapter_name, target, **kwargs)
        if adapter_name not in self.active_adapters:
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)
    else:
        target.update_layer(
            adapter_name,
            r=miss_config.r,
            init_weights=miss_config.init_weights,
            miss_dropout=miss_config.miss_dropout,
            mini_r=miss_config.mini_r,
        )
```

**Flow:**
1. Extract configuration parameters
2. Check if target already has MissLayer
3. If new: create MissLinear wrapper
4. If existing: update with new adapter
5. Handle adapter activation state

### Module Factory

**Method:** `_create_new_module()`

Factory method for creating MissLinear instances:

```python
@staticmethod
def _create_new_module(miss_config, adapter_name, target, **kwargs):
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        new_module = MissLinear(target, adapter_name, **kwargs)
    else:
        raise ValueError(
            f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
        )

    return new_module
```

**Supported Layers:**
- `torch.nn.Linear`: Creates `MissLinear`
- Unsupported: Conv2d, Embedding, etc.

## Integration Features

### Target Module Mapping

Pre-defined module patterns for popular architectures:

```python
target_module_mapping = TRANSFORMERS_MODELS_TO_MISS_TARGET_MODULES_MAPPING

# Example mappings:
# BERT: ["query", "key", "value"]
# GPT-2: ["c_attn", "c_proj"]
# LLaMA: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Adapter Management

**Multiple Adapters:**
```python
# Add first adapter
model = MissModel(base_model, config1, "adapter1")

# Add second adapter (update existing layers)
model._create_and_replace(config2, "adapter2", ...)

# Switch between adapters
model.set_adapter("adapter1")
model.set_adapter("adapter2")
```

**Adapter Deactivation:**
```python
if adapter_name not in self.active_adapters:
    new_module.requires_grad_(False)
```

## Usage Example

```python
from diffusers import StableDiffusionPipeline
from peft import MissModel, MissConfig

# Configure MiSS for text encoder
config_te = MissConfig(
    r=8,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    init_weights=True,
)

# Configure for UNet
config_unet = MissConfig(
    r=8,
    target_modules=[
        "proj_in", "proj_out", "to_k", "to_q", "to_v",
        "to_out.0", "ff.net.0.proj", "ff.net.2",
    ],
    init_weights=True,
)

# Apply adapters
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.text_encoder = MissModel(model.text_encoder, config_te, "default")
model.unet = MissModel(model.unet, config_unet, "default")
```

## Design Patterns

### Factory Pattern
```python
@staticmethod
def _create_new_module(miss_config, adapter_name, target, **kwargs):
    # Creates appropriate MissLinear based on target type
```

### Strategy Pattern
```python
# Different init modes: True, "bat", "mini"
new_module = MissLinear(target, adapter_name, init_weights=miss_config.init_weights)
```

### Template Method Pattern
```python
class MissModel(BaseTuner):
    # Inherits: inject_adapter, merge_adapter, unload
    # Overrides: _create_and_replace, _create_new_module
```

## References

- **Paper**: https://huggingface.co/papers/2409.15371
- **Type**: `PeftType.MISS`
- **Prefix**: "miss_"
- **Supported Layers**: Linear only
