# VeRA Configuration

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/vera/config.py`
- **Lines**: 162
- **Purpose**: Configuration dataclass for VeRA (Vector-based Random Matrix Adaptation) models

## Overview

This module defines `VeraConfig`, a dataclass that stores all configuration parameters for VeRA adapters. VeRA uses shared random projection matrices with learned scaling vectors, requiring fewer trainable parameters than LoRA while maintaining comparable performance.

## VeraConfig Class

**Inheritance**: Extends `PeftConfig` from `peft.config`

**Purpose**: Stores configuration for VeRA model initialization and training

### Configuration Parameters

#### Core Parameters

1. **`r` (int, default=256)**
   - VeRA rank dimension
   - Typically higher than LoRA ranks due to parameter efficiency
   - Paper recommends values like 128, 256 (see Table 1)
   - Defines size of lambda_d vector

2. **`target_modules` (Union[List[str], str], optional)**
   - Module names or regex patterns to apply VeRA to
   - Examples:
     - List: `['q', 'v']`
     - Regex: `'.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'`
   - Only linear layers are supported
   - Converted to set if provided as list

3. **`projection_prng_key` (int, default=0)**
   - PRNG seed for initializing vera_A and vera_B matrices
   - Used when creating new models or loading checkpoints without projections
   - Must be consistent across all adapters in a model
   - Ensures reproducibility of shared projection matrices

4. **`save_projection` (bool, default=True)**
   - Whether to save vera_A/vera_B in state dict
   - If True: Larger checkpoint, guaranteed reloadability
   - If False: Smaller checkpoint, relies on PRNG key to regenerate matrices
   - Warning issued if False due to portability concerns

#### Adapter Parameters

5. **`vera_dropout` (float, default=0.0)**
   - Dropout probability for VeRA layers
   - Applied before first projection in forward pass

6. **`d_initial` (float, default=0.1)**
   - Initial value for `vera_lambda_d` vector
   - Paper recommends small values (≤0.1, see Table 6c)
   - Controls initial adaptation magnitude

#### Layer Configuration

7. **`fan_in_fan_out` (bool, default=False)**
   - Set True if layer stores weights as (fan_in, fan_out)
   - Example: GPT-2's Conv1D uses this format
   - Affects weight transpose operations

8. **`bias` (str, default="none")**
   - Bias update strategy
   - Options:
     - `"none"`: No bias updates
     - `"all"`: Update all biases
     - `"vera_only"`: Update only VeRA layer biases
   - Note: Non-"none" values mean disabled adapters still differ from base model

9. **`modules_to_save` (List[str], optional)**
   - Additional modules to train and save (beyond VeRA layers)
   - Useful for task-specific heads (e.g., classifier, score layers)
   - Example: `["classifier"]` for sequence classification

10. **`init_weights` (bool, default=True)**
    - Whether to initialize VeRA parameters with defaults
    - Should not be changed unless you know exactly what you're doing

#### Selective Layer Application

11. **`layers_to_transform` (Union[List[int], int], optional)**
    - Layer indices to apply VeRA transformations
    - Examples:
      - Single layer: `5`
      - Multiple layers: `[0, 2, 4, 6]`
    - If None, applies to all matching target_modules

12. **`layers_pattern` (Union[List[str], str], optional)**
    - Pattern name for the nn.ModuleList to target
    - Common patterns: `"layers"`, `"h"`
    - Only used with `layers_to_transform`
    - Must be specified if `layers_to_transform` is provided

### Post-Initialization (`__post_init__`)

Performs validation and setup after dataclass initialization:

```python
def __post_init__(self):
    super().__post_init__()
    self.peft_type = PeftType.VERA

    # Convert target_modules list to set
    self.target_modules = (
        set(self.target_modules) if isinstance(self.target_modules, list)
        else self.target_modules
    )

    # Validate layers_pattern and layers_to_transform
    if self.layers_pattern and not self.layers_to_transform:
        raise ValueError(
            "When `layers_pattern` is specified, "
            "`layers_to_transform` must also be specified."
        )

    # Warn about save_projection=False
    if not self.save_projection:
        warnings.warn(
            "Specified to not save vera_A and vera_B within the state dictionary, "
            "instead they will be restored using the PRNG key stored in "
            "`config.projection_prng_key`. Consider setting `config.save_projection` "
            "to `True` to guarantee restoring the checkpoint correctly on all "
            "system configurations."
        )
```

## Configuration Patterns

### Basic Configuration
```python
from peft import VeraConfig

config = VeraConfig(
    r=256,
    target_modules=["q_proj", "v_proj"],
    vera_dropout=0.1,
    d_initial=0.1
)
```

### Full Model Adaptation
```python
config = VeraConfig(
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    vera_dropout=0.05,
    bias="none"
)
```

### Selective Layer Application
```python
config = VeraConfig(
    r=256,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[0, 2, 4, 6, 8, 10],  # Only even layers
    layers_pattern="layers"
)
```

### Lightweight Checkpoint
```python
config = VeraConfig(
    r=256,
    target_modules=["q_proj", "v_proj"],
    save_projection=False,  # Smaller checkpoints
    projection_prng_key=42  # For reproducibility
)
```

### With Task-Specific Head
```python
config = VeraConfig(
    r=256,
    target_modules=["q_proj", "v_proj"],
    modules_to_save=["classifier", "score"],  # Train these too
    bias="vera_only"
)
```

## Design Considerations

### 1. Rank Selection
- VeRA typically uses higher ranks than LoRA (128-256 vs 8-64)
- Higher ranks possible due to dramatically reduced trainable parameters
- Refer to paper Table 1 for guidance

### 2. Shared Projection Strategy
- `projection_prng_key` must match across all adapters
- `save_projection` affects checkpoint size vs portability tradeoff
- Consistent key ensures reproducible shared matrices

### 3. Initialization
- `d_initial=0.1` recommended (paper Table 6c)
- Small values prevent large initial adaptations
- Lambda_b initialized to zeros, lambda_d to d_initial

### 4. Layer Compatibility
- Only supports Linear and Conv1D layers
- `fan_in_fan_out` must match layer weight layout
- Automatic validation during model creation

## Validation Rules

1. **layers_pattern + layers_to_transform**: Both must be specified together or neither
2. **target_modules**: Must be set (no default)
3. **projection_prng_key**: Must be consistent across all adapters in a model
4. **save_projection**: Warning issued if False

## Integration with PEFT

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, VeraConfig

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Create config
config = VeraConfig(
    r=256,
    target_modules=["q_proj", "v_proj"],
    vera_dropout=0.1
)

# Apply VeRA
model = get_peft_model(base_model, config)
```

## Configuration Storage

Configuration is saved in adapter_config.json:
```json
{
  "peft_type": "VERA",
  "r": 256,
  "target_modules": ["q_proj", "v_proj"],
  "projection_prng_key": 0,
  "save_projection": true,
  "vera_dropout": 0.1,
  "d_initial": 0.1,
  "fan_in_fan_out": false,
  "bias": "none",
  "modules_to_save": null
}
```

## References

- **Paper**: https://huggingface.co/papers/2310.11454
- **VeRA**: Dawid J. Kopiczko et al., "VeRA: Vector-based Random Matrix Adaptation"
- Recommends r=128 or 256 with d_initial≤0.1 for best results
