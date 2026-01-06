# src/peft/tuners/lokr/model.py

## Overview
Implementation of the LoKr (Low-Rank Kronecker Product) model class that applies Kronecker product-based parameter-efficient fine-tuning to neural networks. This class manages the creation and injection of LoKr adapters into target layers.

## Class: LoKrModel

Inherits from `LycorisTuner` and provides LoKr-specific adapter management.

### Class Attributes

- **prefix**: "lokr_" - Prefix used for LoKr parameter names
- **tuner_layer_cls**: LoKrLayer - Base layer class for adapters
- **target_module_mapping**: TRANSFORMERS_MODELS_TO_LOKR_TARGET_MODULES_MAPPING - Default target modules per architecture
- **layers_mapping**: Dictionary mapping PyTorch layer types to LoKr implementations:
  - `torch.nn.Conv2d` → `Conv2d`
  - `torch.nn.Conv1d` → `Conv1d`
  - `torch.nn.Linear` → `Linear`

### Methods

#### _create_and_replace()
Private method that creates and replaces target modules with LoKr adapters.

**Parameters:**
- **config**: LycorisConfig with LoKr settings
- **adapter_name**: Name for the adapter
- **target**: The module to be adapted
- **target_name**: Name of the target module
- **parent**: Parent module containing the target
- **current_key**: Full key path to the module

**Process:**
1. Get rank pattern key from config using current_key
2. Get alpha pattern key from config using current_key
3. Extract all config settings as kwargs
4. Override rank and alpha with pattern-specific values if found
5. Add rank_dropout_scale to kwargs
6. If target is already a LoKrLayer:
   - Call `target.update_layer()` to add new adapter
7. If target is a regular layer:
   - Create new LoKr layer using `_create_new_module()`
   - Replace old module with new LoKr-wrapped module

**Pattern Matching:**
The method supports per-layer customization via rank_pattern and alpha_pattern:
- Patterns are regex expressions matched against module keys
- Allows different ranks/alphas for different parts of the model
- Falls back to default r and alpha if no pattern matches

## Architecture Support

### Automatic Target Module Detection
LoKr includes built-in support for various transformer architectures through `TRANSFORMERS_MODELS_TO_LOKR_TARGET_MODULES_MAPPING`. When `target_modules` is not specified in config, it automatically selects appropriate modules based on model type.

### Layer Type Support
LoKr supports three main layer types:
1. **Linear layers**: Most common in transformers (attention, FFN)
2. **Conv2d layers**: Used in vision models and some hybrid architectures
3. **Conv1d layers**: Used in some sequence models

## Usage Example

As shown in the docstring:

```python
from diffusers import StableDiffusionPipeline
from peft import LoKrModel, LoKrConfig

# Configure for text encoder
config_te = LoKrConfig(
    r=8,
    lora_alpha=32,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
)

# Configure for UNet (with effective conv2d)
config_unet = LoKrConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "proj_in", "proj_out", "to_k", "to_q", "to_v",
        "to_out.0", "ff.net.0.proj", "ff.net.2",
    ],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
    use_effective_conv2d=True,  # Enable for conv layers
)

model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.text_encoder = LoKrModel(model.text_encoder, config_te, "default")
model.unet = LoKrModel(model.unet, config_unet, "default")
```

## Design Features

### Inheritance from LycorisTuner
LoKrModel inherits from LycorisTuner, which provides:
- Base adapter management functionality
- Module injection and replacement logic
- State dict handling
- Adapter merging/unmerging

### Per-Layer Customization
Through rank_pattern and alpha_pattern:
- Different layers can have different ranks
- Allows for layer-specific parameter budgets
- Useful for focusing adaptation on specific model parts

### Multi-Adapter Support
Like other PEFT methods, LoKr supports multiple adapters:
- Multiple adapters can coexist on the same model
- Adapters can be switched or combined
- Useful for multi-task scenarios

## Implementation Details

### Module Discovery
The model uses the `layers_mapping` dictionary to determine which LoKr implementation to use based on the base layer type. This allows automatic handling of different layer architectures without manual specification.

### Pattern-Based Configuration
The rank and alpha patterns use `get_pattern_key()` utility to match regex patterns against module keys. This provides flexible, expressive configuration for complex models where different parts need different adaptation strengths.

## References
- Original LoKr concept: https://huggingface.co/papers/2108.06098
- Extended implementation: https://huggingface.co/papers/2309.14859
- LyCORIS implementation: https://github.com/KohakuBlueleaf/LyCORIS

## Integration
This class is instantiated by PEFT's `get_peft_model()` function when using LoKrConfig, or can be used directly for more control over the adaptation process.
