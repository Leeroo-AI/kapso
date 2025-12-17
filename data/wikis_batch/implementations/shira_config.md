# SHiRA Configuration

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/shira/config.py`
- **Lines**: 129
- **Purpose**: Configuration dataclass for SHiRA (Sparse High Rank Adapter) models

## Overview

This module defines `ShiraConfig`, a dataclass that stores configuration parameters for SHiRA adapters. SHiRA uses sparse high-rank matrices instead of dense low-rank matrices, maintaining the same parameter count as LoRA while achieving higher effective rank.

## ShiraConfig Class

**Inheritance**: Extends `PeftConfig` from `peft.config`

**Purpose**: Stores configuration for SHiRA model initialization and training

### Configuration Parameters

#### Core Parameters

1. **`r` (int, default=32)**
   - Parameter budget factor
   - For m×n layer: num_params = r(m + n)
   - Does NOT restrict rank (SHiRA is high-rank)
   - Same parameter count as LoRA with rank r
   - Common values: 8, 16, 32, 64

2. **`mask_type` (Literal["random"], default="random")**
   - Type of sparsity pattern
   - Currently only "random" is built-in
   - Can be extended with custom mask functions

3. **`random_seed` (int, optional, default=None)**
   - Random seed for mask generation (when mask_type="random")
   - Ensures reproducibility of sparse pattern
   - If None, uses random mask each time

4. **`target_modules` (Union[List[str], str], optional)**
   - Module names or regex patterns to apply SHiRA to
   - Examples:
     - List: `['q', 'v']`
     - Regex: `'.*decoder.*(SelfAttention).*(q|v)$'`
   - Only linear layers supported
   - Converted to set if provided as list

#### Layer Configuration

5. **`fan_in_fan_out` (bool, default=False)**
   - Set True if layer stores weights as (fan_in, fan_out)
   - Example: GPT-2's Conv1D
   - Currently not used (only Linear supported)

6. **`init_weights` (bool, default=True)**
   - Initialize sparse weights to zeros
   - If False: Initialize to randn (for testing only)
   - Recommended: True for training

7. **`modules_to_save` (List[str], optional)**
   - Additional modules to train and save
   - Beyond SHiRA layers
   - Example: `["classifier", "score"]`

### Custom Mask Function

The config supports custom mask functions:

```python
config = ShiraConfig(r=32)
config.mask_fn = my_custom_mask_function
```

**Mask Function Signature**:
```python
def mask_fn(base_layer: nn.Module, r: int, **kwargs) -> torch.Tensor:
    """
    Generate binary mask for sparse pattern.

    Args:
        base_layer: The layer to create mask for
        r: Parameter budget factor
        **kwargs: Additional arguments (e.g., random_seed)

    Returns:
        Binary mask tensor (0 or 1)
        Shape: (out_features, in_features)
        num_nonzero = r * (in_features + out_features)
        dtype and device must match base_layer.weight
    """
    pass
```

### Post-Initialization (`__post_init__`)

Performs validation and setup:

```python
def __post_init__(self):
    super().__post_init__()
    self.peft_type = PeftType.SHIRA

    # Convert target_modules to set
    self.target_modules = (
        set(self.target_modules) if isinstance(self.target_modules, list)
        else self.target_modules
    )

    # Set mask function based on mask_type
    if self.mask_type == "random":
        self.mask_fn = random_mask
    else:
        if not self.inference_mode:
            warnings.warn(
                f"Argument {self.mask_type=} is not recognized, "
                "please supply your own masking function by calling "
                "`config.mask_fn = my_mask_fn`."
            )
        self.mask_fn = None
```

## Built-in Mask Functions

### Random Mask

**Implementation** (from mask_functions.py):
```python
def random_mask(base_layer, r, random_seed=None):
    m, n = base_layer.weight.shape
    num_params = r * (m + n)

    total_positions = m * n

    if random_seed is not None:
        generator = torch.Generator().manual_seed(random_seed)
        indices = torch.randperm(total_positions, generator=generator)
    else:
        indices = torch.randperm(total_positions)

    selected = indices[:num_params]

    mask = torch.zeros(m, n, dtype=base_layer.weight.dtype, device=base_layer.weight.device)
    mask.view(-1)[selected] = 1.0

    return mask
```

**Properties**:
- Uniform random selection of sparse positions
- Exactly r(m + n) non-zero elements
- Reproducible with random_seed
- No structured pattern

## Configuration Patterns

### Basic Configuration
```python
from peft import ShiraConfig

config = ShiraConfig(
    r=32,
    target_modules=["q_proj", "v_proj"],
    mask_type="random",
    random_seed=42
)
```

### Full Model Adaptation
```python
config = ShiraConfig(
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    init_weights=True
)
```

### With Custom Mask
```python
def block_sparse_mask(base_layer, r, block_size=16, **kwargs):
    """Block-structured sparse pattern"""
    m, n = base_layer.weight.shape
    num_params = r * (m + n)

    # Implementation...
    return mask

config = ShiraConfig(r=32, target_modules=["q_proj", "v_proj"])
config.mask_fn = block_sparse_mask
```

### Reproducible Random Mask
```python
config = ShiraConfig(
    r=32,
    target_modules=["q_proj", "v_proj"],
    mask_type="random",
    random_seed=42  # Fixed seed for reproducibility
)
```

### With Task-Specific Head
```python
config = ShiraConfig(
    r=32,
    target_modules=["q_proj", "v_proj"],
    modules_to_save=["classifier"],
    init_weights=True
)
```

## Design Considerations

### 1. Parameter Budget vs Rank

**LoRA** (rank r):
- Parameters: r(m + n)
- Effective rank: r

**SHiRA** (budget r):
- Parameters: r(m + n) (same)
- Effective rank: up to min(m, n)
- Sparse structure

**Example** (m=n=4096):
- r=32: Same 262,144 params
- LoRA rank: ≤32
- SHiRA rank: ≤4096 (128x higher)

### 2. Mask Pattern Selection

**Random Mask**:
- Pros: Simple, no assumptions
- Cons: No structure

**Custom Masks**:
- Block-structured: Better for certain hardware
- Magnitude-based: Select important positions
- Learned masks: Train mask selection

### 3. Initialization

**Zero Initialization (Recommended)**:
- Start with identity function
- Gradual learning
- Stable training

**Random Initialization**:
- For testing/debugging
- Not recommended for production

### 4. Reproducibility

Using `random_seed` ensures:
- Same sparse pattern across runs
- Reproducible results
- Consistent checkpoints

## Parameter Efficiency Examples

### Example 1: Small Model (768 hidden)

**Configuration**:
```python
r = 16
m = n = 768
```

**Per Layer**:
- Parameters: 16 × (768 + 768) = 24,576
- Sparsity: 24,576 / (768 × 768) = 4.17%
- Effective rank: up to 768

**vs LoRA**:
- Same parameters: 24,576
- LoRA rank: ≤16
- SHiRA rank: ≤768 (48x higher)

### Example 2: Large Model (4096 hidden)

**Configuration**:
```python
r = 32
m = n = 4096
```

**Per Layer**:
- Parameters: 32 × (4096 + 4096) = 262,144
- Sparsity: 262,144 / (4096 × 4096) = 1.56%
- Effective rank: up to 4096

**vs LoRA**:
- Same parameters: 262,144
- LoRA rank: ≤32
- SHiRA rank: ≤4096 (128x higher)

### Example 3: Full 7B Model

**Model**: 32 layers, 4 attention modules each

**Configuration**:
```python
r = 32
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

**Total**:
- Layers: 32 × 4 = 128
- Params per layer: 262,144
- Total: 128 × 262,144 = 33,554,432 (~34M)

**vs LoRA** (same r=32):
- Same total parameters: ~34M
- But much higher effective rank per layer

## Validation Rules

1. **r > 0**: Must be positive integer
2. **target_modules**: Must be set (no default)
3. **num_params ≤ total_params**: r(m+n) ≤ m×n
   - For m=n: r ≤ n (reasonable constraint)
4. **mask_fn must return**:
   - Binary tensor (0 or 1)
   - Correct shape (m × n)
   - Exact number of non-zeros: r(m + n)
   - Matching dtype and device

## Integration with PEFT

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, ShiraConfig

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Create config
config = ShiraConfig(
    r=32,
    target_modules=["q_proj", "v_proj"],
    mask_type="random",
    random_seed=42
)

# Apply SHiRA
model = get_peft_model(base_model, config)

# Train
model.train()
```

## Custom Mask Examples

### 1. Magnitude-Based Mask
```python
def magnitude_mask(base_layer, r, **kwargs):
    """Select positions with largest absolute values"""
    m, n = base_layer.weight.shape
    num_params = r * (m + n)

    # Get weight magnitudes
    magnitudes = base_layer.weight.abs()

    # Flatten and select top-k
    flat_mag = magnitudes.view(-1)
    _, indices = torch.topk(flat_mag, num_params)

    # Create mask
    mask = torch.zeros(m, n, dtype=base_layer.weight.dtype, device=base_layer.weight.device)
    mask.view(-1)[indices] = 1.0

    return mask
```

### 2. Block-Structured Mask
```python
def block_mask(base_layer, r, block_size=16, **kwargs):
    """Select random blocks"""
    m, n = base_layer.weight.shape
    num_params = r * (m + n)

    blocks_m = m // block_size
    blocks_n = n // block_size
    block_params = block_size * block_size

    num_blocks = num_params // block_params

    # Select random blocks
    total_blocks = blocks_m * blocks_n
    selected = torch.randperm(total_blocks)[:num_blocks]

    mask = torch.zeros(m, n, dtype=base_layer.weight.dtype, device=base_layer.weight.device)

    for block_idx in selected:
        i = (block_idx // blocks_n) * block_size
        j = (block_idx % blocks_n) * block_size
        mask[i:i+block_size, j:j+block_size] = 1.0

    return mask
```

### 3. Row-Column Mask
```python
def row_col_mask(base_layer, r, **kwargs):
    """Select full rows and columns"""
    m, n = base_layer.weight.shape

    # Number of rows and columns to select
    # Solve: num_rows * n + num_cols * m = r(m + n)
    num_rows = r // 2
    num_cols = r - num_rows

    mask = torch.zeros(m, n, dtype=base_layer.weight.dtype, device=base_layer.weight.device)

    # Random rows
    row_indices = torch.randperm(m)[:num_rows]
    mask[row_indices, :] = 1.0

    # Random columns
    col_indices = torch.randperm(n)[:num_cols]
    mask[:, col_indices] = 1.0

    return mask
```

## Configuration Storage

Saved in adapter_config.json:
```json
{
  "peft_type": "SHIRA",
  "r": 32,
  "mask_type": "random",
  "random_seed": 42,
  "target_modules": ["q_proj", "v_proj"],
  "fan_in_fan_out": false,
  "init_weights": true,
  "modules_to_save": null
}
```

**Note**: Custom mask functions are not saved in config. They must be reapplied when loading.

## Best Practices

1. **Start with r=32**: Good balance for most models
2. **Use random_seed**: Ensures reproducibility
3. **Init to zeros**: Stable training start
4. **Test custom masks**: Validate before full training
5. **Consider sparsity**: r(m+n)/(m×n) should be small (< 10%)

## Limitations

1. **Custom Mask Persistence**: Not saved in config
2. **Linear Only**: Only Linear layers supported
3. **Mask Regeneration**: Must regenerate on load if using random without seed
4. **No Dynamic Masking**: Mask fixed after initialization

## References

- **Concept**: Sparse high-rank alternative to dense low-rank adaptation
- **Parameter Efficiency**: Same as LoRA but higher effective rank
- **Flexibility**: Customizable sparse patterns via mask functions
