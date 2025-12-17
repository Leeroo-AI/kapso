# VBLoRA Model Implementation

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/vblora/model.py`
- **Lines**: 209
- **Purpose**: VBLoRA model wrapper and orchestration logic

## Overview

This module implements `VBLoRAModel`, which manages the application of VBLoRA (Vector Bank LoRA) to pretrained transformer models. It handles vector bank initialization, layer wrapping, and provides specialized methods for counting savable parameters based on storage strategy.

## VBLoRAModel Class

**Inheritance**: Extends `BaseTuner` from `peft.tuners.tuners_utils`

**Class Attributes**:
- `prefix`: `"vblora_"` - Parameter name prefix
- `tuner_layer_cls`: `VBLoRALayer` - Base layer class
- `target_module_mapping`: `TRANSFORMERS_MODELS_TO_VBLORA_TARGET_MODULES_MAPPING` - Default target mappings

## Core Methods

### 1. `_init_vblora_vector_bank(config, adapter_name)`

**Purpose**: Initializes shared vector bank for an adapter

**Implementation**:
```python
def _init_vblora_vector_bank(self, config: VBLoRAConfig, adapter_name: str) -> None:
    # Create vector bank tensor
    vblora_vector_bank = torch.zeros(config.num_vectors, config.vector_length)

    # Initialize with uniform distribution
    torch.nn.init.uniform_(
        vblora_vector_bank,
        -config.init_vector_bank_bound,
        config.init_vector_bank_bound
    )

    # Store in ParameterDict
    self.vblora_vector_bank[adapter_name] = vblora_vector_bank
```

**Key Points**:
- One vector bank per adapter
- Uniform initialization in range [-bound, +bound]
- Default bound: 0.02
- Shared across all layers of the adapter

**Why Uniform Initialization?**
- Prevents zero gradients (all-zero initialization would fail)
- Small values prevent large initial adaptations
- Provides diverse initial vector set

### 2. `_pre_injection_hook(model, config, adapter_name)`

**Purpose**: Hook called before adapter injection

**Implementation**:
```python
def _pre_injection_hook(self, model: nn.Module, config: VBLoRAConfig, adapter_name: str) -> None:
    # Initialize ParameterDict for vector banks
    self.vblora_vector_bank = nn.ParameterDict({})
```

**When Called**: Before creating any adapter layers

**Note**: Actual vector bank creation happens in `_create_and_replace` (per-layer basis)

### 3. `_create_and_replace(vblora_config, adapter_name, target, target_name, parent, current_key)`

**Purpose**: Creates VBLoRA layer and replaces target module

**Algorithm**:
```python
def _create_and_replace(self, vblora_config, adapter_name, target, target_name, parent, current_key):
    1. Validate current_key is not None

    2. Extract bias configuration:
       bias = hasattr(target, "bias") and target.bias is not None

    3. Build kwargs:
       kwargs = {
           "fan_in_fan_out": vblora_config.fan_in_fan_out,
           "bias": bias,
       }

    4. Initialize vector bank (per adapter):
       self._init_vblora_vector_bank(vblora_config, adapter_name)

    5. If target is already VBLoRA Linear:
       target.update_layer(
           adapter_name=adapter_name,
           vblora_vector_bank=self.vblora_vector_bank,
           r=vblora_config.r,
           topk=vblora_config.topk,
           num_vectors=vblora_config.num_vectors,
           vector_length=vblora_config.vector_length,
           vblora_dropout=vblora_config.vblora_dropout,
           init_logits_std=vblora_config.init_logits_std,
       )

    6. Else:
       new_module = _create_new_module(...)
       if adapter_name not in active_adapter:
           new_module.requires_grad_(False)
       _replace_module(parent, target_name, new_module, target)
```

**Quantization Note**: TODO comment indicates quantization support planned but not yet implemented

### 4. `_create_new_module(vblora_config, vblora_vector_bank, adapter_name, target, **kwargs)` (static)

**Purpose**: Factory method for creating appropriate VBLoRA layer type

**Layer Type Handling**:

1. **BaseTunerLayer**: Extract base layer
   ```python
   if isinstance(target, BaseTunerLayer):
       target_base_layer = target.get_base_layer()
   else:
       target_base_layer = target
   ```

2. **Standard Linear**:
   ```python
   if isinstance(target_base_layer, torch.nn.Linear):
       if kwargs["fan_in_fan_out"]:
           warnings.warn(
               "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
               "Setting fan_in_fan_out to False."
           )
           kwargs["fan_in_fan_out"] = vblora_config.fan_in_fan_out = False
   ```

3. **Conv1D**:
   ```python
   elif isinstance(target_base_layer, Conv1D):
       kwargs["is_target_conv_1d_layer"] = True
       if not kwargs["fan_in_fan_out"]:
           warnings.warn(
               "fan_in_fan_out is set to False but the target module is `Conv1D`. "
               "Setting fan_in_fan_out to True."
           )
           kwargs["fan_in_fan_out"] = vblora_config.fan_in_fan_out = True
   ```

4. **Create VBLoRA Linear**:
   ```python
   new_module = Linear(
       base_layer=target,
       vblora_vector_bank=vblora_vector_bank,
       adapter_name=adapter_name,
       r=vblora_config.r,
       num_vectors=vblora_config.num_vectors,
       vector_length=vblora_config.vector_length,
       topk=vblora_config.topk,
       vblora_dropout=vblora_config.vblora_dropout,
       init_logits_std=vblora_config.init_logits_std,
       **kwargs,
   )
   ```

**Error Handling**: Raises ValueError for unsupported layer types (only Linear and Conv1D supported)

### 5. `get_nb_savable_parameters(adapter="default")`

**Purpose**: Calculates number of parameters that will be saved to checkpoint

**Returns**: `tuple[int, int]` - (vblora_params, other_params)

**Implementation**:
```python
def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
    logits_params = 0
    vector_bank_params = 0
    other_params = 0

    # Count all parameters
    for name, param in self.named_parameters():
        if "vblora_logits" in name:
            logits_params += param.numel()
        elif "vblora_vector_bank" in name:
            vector_bank_params += param.numel()
        elif param.requires_grad:
            other_params += param.numel()

    # Calculate based on storage strategy
    if self.peft_config[adapter].save_only_topk_weights:
        # Optimized storage calculation
        num_vectors = self.peft_config[adapter].num_vectors
        topk = self.peft_config[adapter].topk

        # Determine index dtype based on num_vectors
        if num_vectors < 2**8:
            factor = 0.25  # uint8
        elif num_vectors < 2**15:
            factor = 0.5   # uint16
        elif num_vectors < 2**31:
            factor = 1     # uint32
        else:
            factor = 2     # uint64

        # Calculate optimized storage
        # Weights: (topk - 1) per position (last inferred from softmax)
        topk_weight_params = (
            logits_params / num_vectors * (topk - 1)
        )

        # Indices: topk per position, scaled by dtype factor
        topk_indices_params = (
            logits_params / num_vectors * topk * factor
        )

        vblora_params = int(vector_bank_params + topk_weight_params + topk_indices_params)
    else:
        # Standard storage: all logits + vector bank
        vblora_params = vector_bank_params + logits_params

    return vblora_params, other_params
```

**Storage Calculation Details**:

**Standard Mode** (save_only_topk_weights=False):
- All logits stored: `r × in_tiles × num_vectors + out_tiles × r × num_vectors`
- Vector bank: `num_vectors × vector_length`
- Total: `logits_params + vector_bank_params`

**Optimized Mode** (save_only_topk_weights=True):
- Per logit position:
  - Top-k indices: `topk` indices (dtype based on num_vectors)
  - Top-k weights: `topk - 1` weights (last inferred via softmax constraint)
- Calculation:
  ```
  positions = logits_params / num_vectors
  indices_size = positions × topk × dtype_factor
  weights_size = positions × (topk - 1)
  total = vector_bank + indices_size + weights_size
  ```

**Index Dtype Selection**:
| num_vectors Range | Dtype  | Factor |
|------------------|--------|--------|
| < 256            | uint8  | 0.25   |
| < 32,768         | uint16 | 0.5    |
| < 2^31           | uint32 | 1.0    |
| >= 2^31          | uint64 | 2.0    |

### 6. `print_savable_parameters()`

**Purpose**: Prints parameter counts to console

**Implementation**:
```python
def print_savable_parameters(self) -> None:
    vblora_params, other_params = self.get_nb_savable_parameters()
    print(
        f"VB-LoRA params to-be-saved (float32-equivalent): {vblora_params:,d} "
        f"|| total params to-be-saved: {(vblora_params + other_params):,d}"
    )
```

**Example Output**:
```
VB-LoRA params to-be-saved (float32-equivalent): 98,560 || total params to-be-saved: 98,560
```

**Note**: Reports "float32-equivalent" parameters accounting for reduced index precision

## VBLoRA Application Workflow

```
1. User creates VBLoRAConfig
   └─> Specifies r, num_vectors, vector_length, topk, etc.

2. get_peft_model(base_model, vblora_config)
   └─> Creates VBLoRAModel instance

3. VBLoRAModel.__init__()
   └─> Calls inject_adapter()

4. inject_adapter()
   ├─> _pre_injection_hook()
   │   └─> Initializes empty vblora_vector_bank ParameterDict
   │
   ├─> For each target module:
   │   └─> _create_and_replace()
   │       ├─> _init_vblora_vector_bank() (per adapter)
   │       │   └─> Creates and initializes vector bank
   │       ├─> Creates new VBLoRA layer
   │       └─> Replaces original module
   │
   └─> All layers share same vector bank per adapter

5. Model ready for training/inference
```

## Parameter Counting Examples

### Example 1: Standard Storage

**Configuration**:
```python
r = 4
num_vectors = 60
vector_length = 256
topk = 2
save_only_topk_weights = False

# Model: 4096 hidden size, 32 layers, 4 attention layers each
```

**Calculations** (per layer, 4096 → 4096):
- in_tiles = out_tiles = 4096 / 256 = 16
- logits_A: 4 × 16 × 60 = 3,840
- logits_B: 16 × 4 × 60 = 3,840
- Per layer logits: 7,680

**Total**:
- Logits: 7,680 × 32 × 4 = 983,040
- Vector bank: 60 × 256 = 15,360
- **Total**: 998,400 parameters

### Example 2: Optimized Storage

**Same configuration with save_only_topk_weights=True**:

**Calculations**:
- Logit positions: 983,040 / 60 = 16,384
- Index dtype: num_vectors=60 < 256, so uint8 (factor=0.25)
- Indices: 16,384 × 2 × 0.25 = 8,192 (float32-equivalent)
- Weights: 16,384 × (2-1) = 16,384
- Vector bank: 15,360
- **Total**: 39,936 parameters (~96% reduction)

### Example 3: Comparison with LoRA

**VBLoRA** (from Example 1):
- 998,400 parameters

**LoRA** (same r=4, same layers):
- Per layer: 4 × (4096 + 4096) = 32,768
- Total: 32,768 × 32 × 4 = 4,194,304
- **VBLoRA is ~4x smaller**

**VBLoRA Optimized** (from Example 2):
- 39,936 parameters
- **~100x smaller than LoRA**

## Storage Strategy Decision Flow

```
Training Phase:
└─> save_only_topk_weights = False
    └─> Full logits stored
    └─> Can resume training
    └─> Larger checkpoints

After Training Complete:
└─> save_only_topk_weights = True
    └─> Save final checkpoint
    └─> ~90-96% size reduction
    └─> Inference/merge only
    └─> Cannot resume training
```

## Design Patterns

### 1. Per-Adapter Vector Bank
Each adapter gets its own vector bank:
```python
self.vblora_vector_bank[adapter_name] = ...
```
Allows different adapters to have different vector sets

### 2. Lazy Bank Initialization
Vector bank created during layer replacement, not during pre-hook

### 3. Storage Strategy Abstraction
`get_nb_savable_parameters` abstracts complex storage calculation

### 4. Factory Pattern
`_create_new_module` dispatches to appropriate layer type

## Integration with PEFT

### Basic Usage
```python
from transformers import AutoModelForCausalLM
from peft import VBLoRAConfig, get_peft_model

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Create VBLoRA config
config = VBLoRAConfig(
    r=4,
    num_vectors=60,
    vector_length=256,
    topk=2,
    target_modules=["q_proj", "v_proj"],
    save_only_topk_weights=False
)

# Apply VBLoRA
model = get_peft_model(base_model, config)

# Check parameter count
model.print_savable_parameters()
# Output: VB-LoRA params to-be-saved (float32-equivalent): 98,560 || total params to-be-saved: 98,560
```

### Training to Deployment Workflow
```python
# 1. Train with full storage
config = VBLoRAConfig(
    r=4,
    num_vectors=60,
    vector_length=256,
    topk=2,
    target_modules=["q_proj", "v_proj"],
    save_only_topk_weights=False  # Full storage
)
model = get_peft_model(base_model, config)

# 2. Train
trainer.train()

# 3. Save with full logits (can resume)
model.save_pretrained("checkpoint_full")

# 4. Convert to optimized storage for deployment
config.save_only_topk_weights = True
model.save_pretrained("checkpoint_optimized")

# 5. Load optimized checkpoint for inference
from peft import AutoPeftModel
model = AutoPeftModel.from_pretrained("checkpoint_optimized")
# Can use for inference/merge, but NOT for resuming training
```

### Multi-Adapter Setup
```python
# Add first adapter
model = get_peft_model(base_model, config1)

# Add second adapter
model.add_adapter("task2", config2)

# Each adapter has its own vector bank
# model.vblora_vector_bank["default"] != model.vblora_vector_bank["task2"]
```

## Limitations and Considerations

1. **No Quantization Support Yet**: TODO comment indicates 8-bit/4-bit support planned
2. **Layer Compatibility**: Only Linear and Conv1D layers supported
3. **Training Resumption**: Cannot resume from save_only_topk_weights checkpoints
4. **Vector Length Constraint**: Must divide layer dimensions evenly
5. **Memory**: Vector bank loaded to GPU during training

## Future Enhancements

From TODO comments:
```python
# TODO: add quantization support
```

Likely additions:
- 8-bit quantized VBLoRA (Linear8bitLt)
- 4-bit quantized VBLoRA (Linear4bit)
- Similar to VeRA's bnb integration

## References

- **Paper**: https://huggingface.co/papers/2405.15179
- **Key Innovation**: Shared vector bank with top-k selection
- **Storage Optimization**: save_only_topk_weights provides ~90-96% checkpoint size reduction
- **Recommended Settings**: topk=2, num_vectors=60-256 depending on model size
