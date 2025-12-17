# C3A Configuration

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/c3a/config.py`
- **Lines**: 137
- **Purpose**: Configuration dataclass for C3A (Circular Convolution Adapter) models

## Overview

C3AConfig defines configuration for C3A adapters that use FFT-based circulant matrices. The key parameter is `block_size`, which determines how the weight matrix is divided into circulant blocks.

## C3AConfig Class

**Inheritance**: Extends `PeftConfig`

### Configuration Parameters

1. **`block_size` (int, default=256)**
   - Size of circulant blocks
   - Must divide both input and output dimensions
   - Larger values = fewer parameters
   - Common values: 128, 256, 512, 1024
   - Tip: Use GCD of all target layer dimensions if unsure

2. **`target_modules` (Union[List[str], str], optional)**
   - Modules to apply C3A to
   - Examples: `['q', 'v']`, `'.*attention.*'`
   - Converted to set if list

3. **`bias` (str, default="none")**
   - Bias handling: "none", "all", or "c3a_only"

4. **`modules_to_save` (List[str], optional)**
   - Additional modules to train beyond C3A layers

5. **`layers_to_transform` (Union[List[int], int], optional)**
   - Specific layer indices to transform
   - Cannot be used with regex target_modules

6. **`layers_pattern` (Union[List[str], str], optional)**
   - Pattern name for nn.ModuleList
   - Common: "layers", "h"
   - Requires layers_to_transform

7. **`block_size_pattern` (dict, optional)**
   - Custom block sizes for specific layers
   - Maps layer name/regex to block_size
   - Example: `{"model.decoder.layers.0.encoder_attn.k_proj": 1280}`

8. **`init_weights` (Union[bool, Literal["gaussian", "kaiming_uniform", "xavier_uniform"]], default="xavier_uniform")**
   - Weight initialization method
   - True: zeros (no-op initially)
   - False or "xavier_uniform": Xavier initialization
   - "gaussian": Normal distribution
   - "kaiming_uniform": Kaiming initialization

### Post-Initialization

```python
def __post_init__(self):
    super().__post_init__()
    self.peft_type = PeftType.C3A

    # Convert to set
    self.target_modules = (
        set(self.target_modules) if isinstance(self.target_modules, list)
        else self.target_modules
    )

    # Validate layers_to_transform with target_modules
    if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
        raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

    # Validate layers_pattern with target_modules
    if isinstance(self.target_modules, str) and self.layers_pattern is not None:
        raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
```

## Configuration Patterns

### Basic Configuration
```python
from peft import C3AConfig

config = C3AConfig(
    block_size=256,
    target_modules=["q_proj", "v_proj"]
)
```

### Layer-Specific Block Sizes
```python
config = C3AConfig(
    block_size=256,  # Default
    target_modules=["q_proj", "v_proj", "k_proj"],
    block_size_pattern={
        "k_proj": 512  # k_proj uses larger blocks
    }
)
```

### Full Model Adaptation
```python
config = C3AConfig(
    block_size=256,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
)
```

### Custom Initialization
```python
config = C3AConfig(
    block_size=256,
    target_modules=["q_proj", "v_proj"],
    init_weights="kaiming_uniform"
)
```

### Selective Layer Application
```python
config = C3AConfig(
    block_size=256,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[0, 2, 4, 6],
    layers_pattern="layers"
)
```

## Parameter Efficiency Analysis

### Example: 7B Model (hidden_size=4096)

**Configuration**: block_size=256, targeting 32 attention layers (4 modules each)

**Per Layer** (4096 → 4096):
- Tiles: 4096 / 256 = 16
- Kernel: 16 × 16 × 256 = 65,536 parameters

**Total**:
- Layers: 32 × 4 = 128
- Total: 128 × 65,536 = 8,388,608 (~8.4M)

**vs LoRA** (r=64):
- Per layer: 64 × (4096 + 4096) = 524,288
- Total: 128 × 524,288 = 67,108,864 (~67M)
- **C3A is 8x smaller**

**vs Full Fine-tuning**:
- Per layer: 4096 × 4096 = 16,777,216
- Total: 128 × 16,777,216 = 2,147,483,648 (~2.1B)
- **C3A is 256x smaller**

## Block Size Selection Guide

### For hidden_size=4096

| block_size | Params/Layer | Total (128 layers) | Notes |
|------------|--------------|-------------------|-------|
| 128 | 262,144 | 33.5M | More parameters, higher capacity |
| 256 | 65,536 | 8.4M | Balanced (recommended) |
| 512 | 32,768 | 4.2M | Efficient |
| 1024 | 16,384 | 2.1M | Very efficient, lower capacity |

### For hidden_size=768

| block_size | Params/Layer | Notes |
|------------|--------------|-------|
| 64 | 12,288 | Fine-grained |
| 128 | 6,144 | Balanced |
| 256 | 3,072 | Efficient |

### Selection Criteria

1. **Find GCD**: Greatest common divisor of all target layer dimensions
2. **Consider Powers of 2**: 64, 128, 256, 512, 1024 (FFT efficiency)
3. **Balance**: Smaller = more capacity, larger = fewer params

## Design Considerations

### 1. Block Size Impact

**Small Blocks** (e.g., 64, 128):
- More parameters
- Higher capacity
- More flexible adaptation
- Slower computation

**Large Blocks** (e.g., 512, 1024):
- Fewer parameters
- Lower capacity
- More constrained adaptation
- Faster computation

### 2. Layer-Specific Tuning

Use `block_size_pattern` for heterogeneous layers:
```python
config = C3AConfig(
    block_size=256,
    block_size_pattern={
        "attention": 256,
        "mlp.gate_proj": 512,  # MLP can use larger blocks
        "mlp.up_proj": 512,
        "mlp.down_proj": 512
    }
)
```

### 3. Initialization Strategy

**Zero Initialization** (init_weights=True):
- Start with identity-like behavior
- Gradual adaptation
- Stable training

**Xavier** (default):
- Balanced variance
- Good for most tasks

**Kaiming**:
- ReLU-friendly
- Potentially faster convergence

**Gaussian**:
- Simple random initialization
- May need learning rate tuning

## Validation Rules

1. **block_size > 0**: Must be positive
2. **Divisibility**: block_size must divide target layer dimensions
3. **layers_to_transform**: Cannot use with regex target_modules
4. **layers_pattern**: Requires layers_to_transform
5. **block_size_pattern**: Keys must match layer names

## Integration with PEFT

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, C3AConfig

base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

config = C3AConfig(
    block_size=256,
    target_modules=["q_proj", "v_proj"],
    init_weights="xavier_uniform"
)

model = get_peft_model(base_model, config)
```

## Configuration Storage

```json
{
  "peft_type": "C3A",
  "block_size": 256,
  "target_modules": ["q_proj", "v_proj"],
  "bias": "none",
  "modules_to_save": null,
  "layers_to_transform": null,
  "layers_pattern": null,
  "block_size_pattern": {},
  "init_weights": "xavier_uniform"
}
```

## Best Practices

1. **Start with 256**: Good default for most models
2. **Use GCD**: If unsure, use GCD of layer dimensions
3. **Power of 2**: Ensures FFT efficiency
4. **Test block sizes**: Try 128, 256, 512 to find sweet spot
5. **Layer-specific**: Use block_size_pattern for heterogeneous architectures

## References

- **Paper**: https://huggingface.co/papers/2407.19342
- **Key Innovation**: FFT-based circulant matrices
- **Efficiency**: 8-256x fewer parameters than LoRA/full fine-tuning
