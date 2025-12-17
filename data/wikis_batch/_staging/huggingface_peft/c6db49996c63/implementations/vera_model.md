# VeRA Model Implementation

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/vera/model.py`
- **Lines**: 294
- **Purpose**: VeRA model wrapper and orchestration logic

## Overview

This module implements `VeraModel`, which manages the application of VeRA (Vector-based Random Matrix Adaptation) to pretrained transformer models. It handles shared projection matrix initialization, layer wrapping, and adapter injection while ensuring proper dimension management across all adapted layers.

## Key Components

### Helper Function: `_kaiming_init`

**Purpose**: Kaiming uniform initialization with controllable PRNG

**Signature**:
```python
def _kaiming_init(
    tensor_or_shape: Union[torch.Tensor, tuple[int, ...]],
    generator: torch.Generator,
) -> torch.Tensor
```

**Algorithm**:
```python
1. Create or use tensor
2. Calculate fan_in: fan = _calculate_correct_fan(tensor, "fan_in")
3. Compute gain and std:
   gain = sqrt(2)
   std = gain / sqrt(fan)
   bound = sqrt(3.0) * std
4. Initialize: uniform_(-bound, bound, generator=generator)
```

**Purpose**: Ensures reproducible initialization of vera_A and vera_B matrices using specified PRNG key.

### VeraModel Class

**Inheritance**: Extends `BaseTuner` from `peft.tuners.tuners_utils`

**Class Attributes**:
- `prefix`: `"vera_lambda_"` - Parameter name prefix
- `tuner_layer_cls`: `VeraLayer` - Base layer class
- `target_module_mapping`: `TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING` - Default target mappings

## Core Methods

### 1. `_find_dim(config) -> tuple[int, int]`

**Purpose**: Determines maximum dimensions needed for shared projection matrices

**Algorithm**:
```python
1. Get model config and prepare adapter config
2. Iterate through all modules in model
3. For each target module:
   - If Linear: get (out_features, in_features)
   - If Conv1D: get shape (with ds_shape handling)
4. Track maximum dimensions: largest_shape = max(all_shapes)
5. Return (max_out_features, max_in_features)
```

**Error Handling**:
- Raises ValueError if no compatible layers found
- Suggests checking `peft_config.target_modules`

**Example Output**: `(4096, 4096)` for 7B model with 4096 hidden size

### 2. `_init_vera_A_vera_B(config, adapter_name)`

**Purpose**: Initializes shared frozen projection matrices

**Implementation**:
```python
def _init_vera_A_vera_B(self, config: VeraConfig, adapter_name: str) -> None:
    linear_out_dim, linear_in_dim = self._find_dim(config)

    # Create BufferDict (persistent if save_projection=True)
    self.vera_A = BufferDict({}, persistent=config.save_projection)
    self.vera_B = BufferDict({}, persistent=config.save_projection)

    # Deterministic initialization using PRNG key
    generator = torch.Generator(device="cpu").manual_seed(config.projection_prng_key)
    vera_A = _kaiming_init((config.r, linear_in_dim), generator=generator)
    vera_B = _kaiming_init((linear_out_dim, config.r), generator=generator)

    self.vera_A[adapter_name] = vera_A
    self.vera_B[adapter_name] = vera_B
```

**Key Points**:
- Matrices initialized on CPU
- Kaiming uniform distribution
- Seeded with `projection_prng_key` for reproducibility
- `persistent=config.save_projection` controls state dict inclusion

### 3. `_pre_injection_hook(model, config, adapter_name)`

**Purpose**: Hook called before adapter injection

**Implementation**:
```python
def _pre_injection_hook(self, model: nn.Module, config: VeraConfig, adapter_name: str) -> None:
    self._init_vera_A_vera_B(config, adapter_name)
```

**When Called**: Automatically invoked by BaseTuner before creating adapter layers

### 4. `_check_new_adapter_config(config)`

**Purpose**: Validates configuration when adding new adapters

**Validation Rules**:
```python
1. Call super()._check_new_adapter_config(config)

2. Check projection_prng_key consistency:
   - All adapters must use same key
   - Raises ValueError if mismatch

3. Check save_projection consistency:
   - All adapters must have same setting
   - Raises ValueError if mixed True/False
```

**Example Error**:
```
ValueError: Vera PRNG initialisation key must be the same for all adapters.
Got projection_prng_key=42 but previous config had projection_prng_key=0.
```

### 5. `_create_and_replace(vera_config, adapter_name, target, target_name, parent, current_key, **optional_kwargs)`

**Purpose**: Creates VeRA layer and replaces target module

**Algorithm**:
```python
1. Validate current_key is not None

2. Extract configuration:
   r = vera_config.r
   bias = hasattr(target, "bias") and target.bias is not None

3. Build kwargs dict:
   - r, vera_dropout, fan_in_fan_out, init_weights
   - loaded_in_8bit, loaded_in_4bit flags
   - bias

4. If target is already VeRA Linear:
   target.update_layer(adapter_name, self.vera_A, self.vera_B, ...)
5. Else:
   new_module = _create_new_module(...)
   if adapter_name not in active_adapter:
       new_module.requires_grad_(False)
   _replace_module(parent, target_name, new_module, target)
```

**Quantization Support**: Handles 8-bit and 4-bit quantized models

### 6. `_create_new_module(vera_config, vera_A, vera_B, adapter_name, target, **kwargs)` (static)

**Purpose**: Factory method for creating appropriate VeRA layer type

**Layer Type Handling**:

1. **BaseTunerLayer**: Extract base layer
2. **8-bit Quantized** (bitsandbytes Linear8bitLt):
   ```python
   from .bnb import Linear8bitLt
   return Linear8bitLt(
       target, adapter_name, vera_A, vera_B,
       has_fp16_weights=...,
       threshold=...,
       index=...
   )
   ```

3. **4-bit Quantized** (bitsandbytes Linear4bit):
   ```python
   from .bnb import Linear4bit
   return Linear4bit(
       target, adapter_name, vera_A, vera_B,
       compute_dtype=...,
       compress_statistics=...,
       quant_type=...
   )
   ```

4. **Standard Linear**:
   ```python
   if fan_in_fan_out:
       warnings.warn("fan_in_fan_out=True but target is Linear, setting False")
       vera_config.fan_in_fan_out = False
   return Linear(target, vera_A, vera_B, adapter_name, bias=bias, d_initial=..., **kwargs)
   ```

5. **Conv1D**:
   ```python
   kwargs["is_target_conv_1d_layer"] = True
   if not fan_in_fan_out:
       warnings.warn("fan_in_fan_out=False but target is Conv1D, setting True")
       vera_config.fan_in_fan_out = True
   return Linear(...)
   ```

**Error Handling**: Raises ValueError for unsupported layer types

## VeRA Application Workflow

```
1. User creates VeraConfig
   └─> Specifies r, target_modules, projection_prng_key, etc.

2. get_peft_model(base_model, vera_config)
   └─> Creates VeraModel instance

3. VeraModel.__init__()
   └─> Calls inject_adapter()

4. inject_adapter()
   ├─> _pre_injection_hook()
   │   └─> _init_vera_A_vera_B()
   │       └─> Creates shared vera_A, vera_B matrices
   │
   ├─> For each target module:
   │   └─> _create_and_replace()
   │       ├─> Checks if already VeRA layer
   │       ├─> Creates new VeRA layer if needed
   │       └─> Replaces original module
   │
   └─> All adapted layers share same vera_A, vera_B

5. Model ready for training/inference
```

## Shared Matrix Management

### Dimension Strategy

VeRA uses a "maximum dimensions" strategy:

```python
# Example: Model with varying layer dimensions
layer1: 768 × 768
layer2: 768 × 3072
layer3: 3072 × 768

# Shared matrices sized for maximum
vera_A: (r, 3072)  # max input dimension
vera_B: (3072, r)  # max output dimension

# Each layer slices what it needs
layer1: vera_A[:, :768], vera_B[:768, :]
layer2: vera_A[:, :768], vera_B[:3072, :]
layer3: vera_A[:, :3072], vera_B[:768, :]
```

### Benefits
- Single pair of matrices for entire model
- Minimal memory overhead
- All layers share same random projections

## Configuration Validation

### Multi-Adapter Constraints

When adding multiple adapters:

1. **PRNG Key Consistency**:
   ```python
   model = get_peft_model(base_model, config1)  # key=0
   model.add_adapter("adapter2", config2)       # key must be 0
   # Error if config2.projection_prng_key != 0
   ```

2. **Save Projection Consistency**:
   ```python
   # All adapters must agree on save_projection
   config1.save_projection = True
   config2.save_projection = True  # Must match
   ```

### Rationale
- Shared matrices must be identical across adapters
- Different PRNG keys would create incompatible projections
- Mixed save_projection settings cause state dict inconsistency

## Quantization Support

VeRA supports quantized base models:

### 8-bit Quantization (bitsandbytes)
```python
from transformers import AutoModelForCausalLM
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    load_in_8bit=True,
    device_map="auto"
)

config = VeraConfig(r=256, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
# Automatically uses Linear8bitLt wrapper
```

### 4-bit Quantization
```python
base_model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    load_in_4bit=True,
    device_map="auto"
)

config = VeraConfig(r=256, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
# Automatically uses Linear4bit wrapper
```

## Example Usage

### Basic Application
```python
from transformers import AutoModelForCausalLM
from peft import VeraConfig, get_peft_model

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Create VeRA config
config = VeraConfig(
    r=256,
    target_modules=["q_proj", "v_proj"],
    vera_dropout=0.1,
    d_initial=0.1,
    projection_prng_key=42
)

# Apply VeRA
model = get_peft_model(base_model, config)

# Train model
model.train()
```

### Multi-Adapter Setup
```python
# Add first adapter
model = get_peft_model(base_model, config1)

# Add second adapter (must have same projection_prng_key)
config2 = VeraConfig(
    r=256,
    target_modules=["k_proj", "o_proj"],
    projection_prng_key=42  # Must match config1
)
model.add_adapter("task2", config2)

# Switch between adapters
model.set_adapter("default")  # Use first adapter
model.set_adapter("task2")    # Use second adapter
```

## Design Patterns

### 1. Lazy Initialization
Shared matrices created only once during first adapter injection

### 2. Factory Pattern
`_create_new_module` dispatches to appropriate layer type

### 3. Hook Pattern
`_pre_injection_hook` allows setup before layer creation

### 4. Validation Chain
Configuration checks ensure compatibility across adapters

## References

- **Paper**: https://huggingface.co/papers/2310.11454
- **Related**: LoRA (Low-Rank Adaptation) - parent technique
- **Key Innovation**: Shared random projections with learned scaling vectors
