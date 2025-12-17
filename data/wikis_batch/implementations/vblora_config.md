# VBLoRA Configuration

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/vblora/config.py`
- **Lines**: 196
- **Purpose**: Configuration dataclass for VBLoRA (Vector Bank LoRA) models

## Overview

This module defines `VBLoRAConfig`, a dataclass that stores all configuration parameters for VBLoRA adapters. VBLoRA uses a shared vector bank with top-k selection to construct low-rank adaptation matrices, providing significant parameter savings especially when storing checkpoints.

## VBLoRAConfig Class

**Inheritance**: Extends `PeftConfig` from `peft.config`

**Purpose**: Stores configuration for VBLoRA model initialization and training

### Configuration Parameters

#### Core Adaptation Parameters

1. **`r` (int, default=4)**
   - Rank of incremental matrices
   - Similar to LoRA rank but typically smaller due to vector bank efficiency
   - Defines intermediate dimension in the adaptation

2. **`num_vectors` (int, default=256)**
   - Number of vectors in the shared vector bank
   - Larger models typically need more vectors
   - Recommended: 60-256 depending on model size
   - Shared across all adapted layers

3. **`vector_length` (int, default=256)**
   - Length of each vector in the bank
   - Must be divisible by hidden dimensions of target layers
   - Determines tiling granularity
   - Common values: 128, 256, 512

4. **`topk` (int, default=2)**
   - K value for top-K selection
   - Number of vectors selected per position
   - Paper recommendation: topk=2 for best performance/efficiency
   - Higher values increase checkpoint size

#### Module Selection

5. **`target_modules` (Union[List[str], str], optional)**
   - Module names or regex patterns to apply VBLoRA to
   - Examples:
     - List: `['q_proj', 'v_proj']`
     - Regex: `'.*decoder.*(SelfAttention).*(q|v)$'`
     - Wildcard: `'all-linear'` (all linear layers except output)
   - If not specified, uses architecture-specific defaults
   - Converted to set if provided as list

6. **`exclude_modules` (Union[List[str], str], optional)**
   - Module names or regex patterns to exclude from VBLoRA
   - Useful with `'all-linear'` to exclude specific layers
   - Converted to set if provided as list

#### Storage Optimization

7. **`save_only_topk_weights` (bool, default=False)**
   - **Critical Parameter**: Controls checkpoint storage strategy
   - **If False** (default):
     - Saves full logits tensors
     - Models can be used for training, inference, or merging
     - Larger checkpoint size
   - **If True**:
     - Saves only top-k indices and weights
     - ~90% storage reduction for topk=2
     - Models can ONLY be used for merging or inference
     - **Cannot resume training**
   - Warning: Setting True is one-way (training cannot be resumed)

#### Training Parameters

8. **`vblora_dropout` (float, default=0.0)**
   - Dropout probability for VBLoRA layers
   - Applied before first linear transformation
   - Typically kept at 0.0 for VBLoRA

9. **`init_vector_bank_bound` (float, default=0.02)**
   - Uniform distribution bounds for vector bank initialization
   - Range: [-init_vector_bank_bound, +init_vector_bank_bound]
   - **Important**: Do NOT initialize to all zeros (causes zero gradients)
   - Small positive values (0.01-0.05) work well
   - Large values may cause training instability

10. **`init_logits_std` (float, default=0.1)**
    - Standard deviation for logits initialization
    - Logits initialized as normal(0, init_logits_std)
    - Default 0.1 typically works well
    - Affects initial vector selection entropy

#### Layer Configuration

11. **`fan_in_fan_out` (bool, default=False)**
    - Set True if layer stores weights as (fan_in, fan_out)
    - Example: GPT-2's Conv1D uses this format
    - Affects weight transpose operations

12. **`bias` (str, default="none")**
    - Bias update strategy
    - Options:
      - `"none"`: No bias updates
      - `"all"`: Update all biases
      - `"vblora_only"`: Update only VBLoRA layer biases
    - Note: Non-"none" values mean disabled adapters differ from base model

13. **`modules_to_save` (List[str], optional)**
    - Additional modules to train and save (beyond VBLoRA layers)
    - Useful for task-specific heads
    - Example: `["classifier", "score"]` for classification tasks

#### Selective Layer Application

14. **`layers_to_transform` (Union[List[int], int], optional)**
    - Layer indices to apply VBLoRA transformations
    - Examples:
      - Single layer: `5`
      - Multiple layers: `[0, 2, 4, 6]`
    - If None, applies to all matching target_modules

15. **`layers_pattern` (Union[List[str], str], optional)**
    - Pattern name for the nn.ModuleList to target
    - Common patterns: `"layers"`, `"h"`
    - Only used with `layers_to_transform`

### Post-Initialization (`__post_init__`)

Performs validation and setup:

```python
def __post_init__(self):
    super().__post_init__()
    self.peft_type = PeftType.VBLORA

    # Convert to sets
    self.target_modules = (
        set(self.target_modules) if isinstance(self.target_modules, list)
        else self.target_modules
    )
    self.exclude_modules = (
        set(self.exclude_modules) if isinstance(self.exclude_modules, list)
        else self.exclude_modules
    )

    # Validate layers_pattern and layers_to_transform
    if self.layers_pattern and not self.layers_to_transform:
        raise ValueError(
            "When `layers_pattern` is specified, "
            "`layers_to_transform` must also be specified."
        )
```

## Configuration Patterns

### Basic Configuration
```python
from peft import VBLoRAConfig

config = VBLoRAConfig(
    r=4,
    num_vectors=60,
    vector_length=256,
    topk=2,
    target_modules=["q_proj", "v_proj"]
)
```

### Lightweight Checkpoint (Storage Optimized)
```python
config = VBLoRAConfig(
    r=4,
    num_vectors=60,
    vector_length=256,
    topk=2,
    target_modules=["q_proj", "v_proj"],
    save_only_topk_weights=True  # ~90% smaller checkpoint
)
# Warning: Cannot resume training from this checkpoint
```

### Full Model Adaptation
```python
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    topk=2,
    target_modules="all-linear",
    exclude_modules=["lm_head"]
)
```

### Large Model Configuration
```python
config = VBLoRAConfig(
    r=4,
    num_vectors=256,  # More vectors for larger models
    vector_length=512,  # Larger tiles
    topk=2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
)
```

### With Task-Specific Head
```python
config = VBLoRAConfig(
    r=4,
    num_vectors=60,
    vector_length=256,
    topk=2,
    target_modules=["q_proj", "v_proj"],
    modules_to_save=["classifier"],
    bias="vblora_only"
)
```

### Selective Layer Application
```python
config = VBLoRAConfig(
    r=4,
    num_vectors=60,
    vector_length=256,
    topk=2,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[0, 2, 4, 6, 8, 10],
    layers_pattern="layers"
)
```

## Design Considerations

### 1. Vector Bank Sizing

**num_vectors**:
- Small models (< 1B params): 60-128
- Medium models (1-7B params): 128-256
- Large models (> 7B params): 256+

**vector_length**:
- Must divide all target layer dimensions
- Common choices: 128, 256, 512
- Smaller = more fine-grained, larger = more efficient
- Example: For hidden_size=4096, use 256 (4096/256=16 tiles)

### 2. Top-K Selection

**Recommended: topk=2**
- Paper shows best performance/efficiency tradeoff
- Higher values:
  - Pros: More expressive
  - Cons: Larger checkpoints, especially with save_only_topk_weights

### 3. Storage Strategy

**save_only_topk_weights Decision**:

Choose **False** (default) if:
- Need to resume training
- Checkpoint size not critical
- Want maximum flexibility

Choose **True** if:
- Only need inference/merging
- Storage is limited
- ~90% size reduction needed
- Training is complete

### 4. Initialization

**Vector Bank**:
```python
init_vector_bank_bound = 0.02
# Uniform[-0.02, 0.02]
# Small non-zero values crucial
```

**Logits**:
```python
init_logits_std = 0.1
# Normal(0, 0.1)
# Controls initial selection entropy
```

## Parameter Efficiency Analysis

### Example: 7B Model with hidden_size=4096

**Target**: 32 attention layers (q_proj, k_proj, v_proj, o_proj)

**Configuration**:
```python
r = 4
num_vectors = 60
vector_length = 256
topk = 2
```

**Parameters per Layer** (4096 → 4096):
- Tiles: 4096 / 256 = 16
- Logits_A: 4 × 16 × 60 = 3,840
- Logits_B: 16 × 4 × 60 = 3,840
- Total per layer: 7,680

**Total Parameters**:
- Logits: 7,680 × 32 × 4 = 983,040
- Vector Bank: 60 × 256 = 15,360 (shared)
- **Total**: ~1M trainable parameters

**With save_only_topk_weights (topk=2)**:
- Indices: 2 per position (uint8 if num_vectors < 256)
- Weights: 1 per position (float32, other inferred from softmax)
- Reduction: ~90%
- **Stored**: ~100K parameters

**LoRA Comparison** (same rank=4):
- Per layer: 4 × (4096 + 4096) = 32,768
- Total: 32,768 × 32 × 4 = 4,194,304 (~4M)
- **VBLoRA is ~4x smaller**

## Configuration Validation

### Constraints

1. **vector_length Division**:
   ```python
   if hidden_size % vector_length != 0:
       raise ValueError("vector_length must divide hidden dimensions")
   ```

2. **layers_pattern Dependency**:
   ```python
   if layers_pattern and not layers_to_transform:
       raise ValueError("Must specify layers_to_transform with layers_pattern")
   ```

3. **topk Range**:
   - Must be positive integer
   - Typically ≤ num_vectors
   - Recommended: 2 for optimal performance

## Integration with PEFT

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, VBLoRAConfig

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Create config
config = VBLoRAConfig(
    r=4,
    num_vectors=60,
    vector_length=256,
    topk=2,
    target_modules=["q_proj", "v_proj"],
    save_only_topk_weights=False  # Can resume training
)

# Apply VBLoRA
model = get_peft_model(base_model, config)

# Train
model.train()

# Save (full logits)
model.save_pretrained("checkpoint")

# Later: Convert to storage-optimized
config.save_only_topk_weights = True
model.save_pretrained("checkpoint_optimized")
```

## Configuration Storage

Saved in adapter_config.json:
```json
{
  "peft_type": "VBLORA",
  "r": 4,
  "num_vectors": 60,
  "vector_length": 256,
  "topk": 2,
  "target_modules": ["q_proj", "v_proj"],
  "exclude_modules": null,
  "save_only_topk_weights": false,
  "vblora_dropout": 0.0,
  "init_vector_bank_bound": 0.02,
  "init_logits_std": 0.1,
  "fan_in_fan_out": false,
  "bias": "none",
  "modules_to_save": null
}
```

## Best Practices

1. **Start with defaults**: r=4, num_vectors=60, vector_length=256, topk=2
2. **Scale num_vectors** with model size: 60 (small), 256 (large)
3. **Keep topk=2** unless you have specific needs
4. **Use save_only_topk_weights=True** only for final deployment
5. **Ensure vector_length divides** hidden dimensions
6. **Initialize vector bank** with small non-zero values

## References

- **Paper**: https://huggingface.co/papers/2405.15179
- **Key Finding**: topk=2 provides best performance/efficiency tradeoff
- **Storage**: save_only_topk_weights reduces checkpoint size by ~90%
