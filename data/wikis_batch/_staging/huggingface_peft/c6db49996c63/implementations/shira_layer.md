# SHiRA Layer Implementation

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/shira/layer.py`
- **Lines**: 217
- **Purpose**: SHiRA (Sparse High Rank Adapter) layer implementation using sparse weight matrices

## Overview

This module implements SHiRA adapter layers, a parameter-efficient fine-tuning method that uses high-rank sparse matrices instead of low-rank dense matrices. SHiRA maintains the same parameter count as LoRA (r(m+n) for an m×n matrix) but achieves higher effective rank through sparsity, potentially improving adaptation quality.

## Key Components

### ShiraLayer (Base Class)

**Purpose**: Base layer for SHiRA adapters managing sparse weight matrices and indices

**Attributes**:
- `adapter_layer_names`: `("shira_weight",)` - Trainable sparse weight vectors
- `other_param_names`: `("r", "scaling", "shira_indices")` - Metadata parameters
- `r`: Dict mapping adapter names to their parameter budgets
- `scaling`: Dict mapping adapter names to scaling factors (default: 1.0)
- `shira_weight`: ParameterDict storing sparse weight values (1D vectors)
- `shira_indices`: Dict storing sparse matrix indices (2D coordinate arrays)
- `weight_shape`: Shape of base layer weight (for sparse tensor construction)

**Key Methods**:

1. **`__init__(base_layer, **kwargs)`**
   - Initializes base layer wrapper
   - Extracts in_features and out_features from base layer
   - Only supports nn.Linear layers currently
   - Sets up adapter parameter dictionaries

2. **`update_layer(adapter_name, mask, r, init_weights=True, inference_mode=False)`**
   - Adds or updates a SHiRA adapter
   - **Parameters**:
     - `mask`: Binary mask (shape: out_features × in_features) indicating sparse positions
     - `r`: Parameter budget (num_shira_params = r × (in_features + out_features))
     - `init_weights`: If True, initialize to zeros; if False, use randn
   - **Validation**:
     - r must be positive
     - num_shira_params must not exceed total layer parameters
     - Mask indices must match weight vector length
   - **Creates**:
     - `shira_weight[adapter_name]`: 1D parameter vector
     - `shira_indices[adapter_name]`: 2D index tensor for sparse construction
   - **Scaling**: Initialized to 1.0 (adjustable at inference)

3. **`reset_shira_parameters(adapter_name)`**
   - Resets parameters to zeros
   - Called during initialization if init_weights=True

4. **`set_scale(adapter, scale)`**
   - Sets scaling factor for an adapter
   - Used during inference to adjust adaptation magnitude
   - Ignores if adapter not in layer

### Linear (Implementation Class)

**Purpose**: SHiRA implementation for Linear layers (inherits from nn.Module and ShiraLayer)

**Constructor Parameters**:
- `base_layer`: Original Linear layer to wrap
- `mask`: Binary mask tensor indicating sparse positions
- `adapter_name`: Name of the adapter
- `r`: Parameter budget
- `fan_in_fan_out`: Whether layer stores weights as (fan_in, fan_out)
- `init_weights`: Whether to initialize to zeros (default: True)

**Key Methods**:

1. **`merge(safe_merge=False, adapter_names=None)`**
   - Merges active adapter weights into base weights
   - Constructs sparse delta tensor and adds to base weight
   - **Parameters**:
     - `safe_merge`: If True, checks for NaNs before committing
     - `adapter_names`: List of adapters to merge (None = all active)
   - Updates `merged_adapters` list

2. **`unmerge()`**
   - Removes merged adapter weights from base weights
   - Reverses the merge operation
   - Pops adapters from `merged_adapters` list

3. **`get_delta_weight(adapter)`**
   - Constructs sparse delta weight tensor
   - **Algorithm**:
     ```python
     1. Ensure indices on correct device
     2. Create sparse COO tensor:
        sparse_tensor = torch.sparse_coo_tensor(
            indices=shira_indices[adapter],
            values=shira_weight[adapter] * scaling[adapter],
            size=weight_shape
        )
     3. Return sparse tensor
     ```
   - Returns sparse tensor (not densified)

4. **`forward(x, *args, **kwargs)`**
   - Forward pass with SHiRA adaptation
   - **Logic Flow**:
     ```python
     if disable_adapters:
         if merged: unmerge()
         return base_layer(x)
     elif merged:
         return base_layer(x)  # Already includes adapter
     else:
         # Create modified weight
         new_weight = copy.deepcopy(base_layer.weight.data)
         for each active adapter:
             new_weight += get_delta_weight(adapter)
         # Use modified weight
         result = F.linear(x, new_weight, bias=base_layer.bias)
         return result
     ```
   - **Note**: Uses deepcopy to avoid modifying base layer weight
   - Accumulates multiple adapters additively

## Mathematical Formulation

SHiRA uses sparse high-rank matrices:

```
output = base_layer(x) + F.linear(x, sparse_delta_weight, None)

where sparse_delta_weight:
- Shape: (out_features, in_features)
- Non-zero elements: r(in_features + out_features)
- Element-wise scaling applied
```

### Sparse Matrix Construction

SHiRA stores sparse matrices efficiently:

**Storage**:
- Values: 1D vector of non-zero elements
- Indices: 2×N array of (row, col) coordinates

**Construction**:
```python
sparse_tensor = torch.sparse_coo_tensor(
    indices=[[row0, row1, ...],    # Row indices
             [col0, col1, ...]],   # Column indices
    values=[val0, val1, ...],      # Non-zero values
    size=(out_features, in_features)
)
```

**Example** (4×4 matrix, r=2, total params = 2×(4+4)=16):
```
Mask:
[1 0 1 0]
[0 1 0 1]
[1 0 1 0]
[0 1 0 1]

Indices: [[0,0,1,1,2,2,3,3,0,0,1,1,2,2,3,3],
          [0,2,1,3,0,2,1,3,0,2,1,3,0,2,1,3]]
          (16 positions)

Values: [w0, w1, w2, ..., w15]  (16 learnable parameters)
```

## Parameter Count Comparison

For an m×n layer:

**Full Fine-Tuning**:
- Parameters: m × n

**LoRA** (rank r):
- Parameters: r(m + n)
- Effective rank: r

**SHiRA** (budget r):
- Parameters: r(m + n) (same as LoRA)
- Effective rank: up to min(m, n) (much higher)
- Sparse structure

**Example** (m=n=4096, r=32):
- Full: 16,777,216 params
- LoRA: 32 × 8,192 = 262,144 params, rank ≤ 32
- SHiRA: 32 × 8,192 = 262,144 params, rank ≤ 4096

## Mask Generation

Masks determine sparse pattern. From config, uses `mask_fn`:

### Random Mask (Default)
```python
def random_mask(base_layer, r, random_seed=None):
    m, n = base_layer.weight.shape
    num_params = r * (m + n)

    # Create flat indices
    total_positions = m * n
    if random_seed is not None:
        generator = torch.Generator().manual_seed(random_seed)
        indices = torch.randperm(total_positions, generator=generator)
    else:
        indices = torch.randperm(total_positions)

    # Select first num_params positions
    selected = indices[:num_params]

    # Convert to 2D mask
    mask = torch.zeros(m, n, dtype=base_layer.weight.dtype, device=base_layer.weight.device)
    mask.view(-1)[selected] = 1.0

    return mask
```

### Custom Masks
Users can provide custom mask functions:
```python
def my_mask_fn(base_layer, r, **kwargs):
    m, n = base_layer.weight.shape
    num_params = r * (m + n)

    # Custom logic to create binary mask
    mask = create_my_sparse_pattern(m, n, num_params)

    # Must be binary (0 or 1)
    # Must have exactly num_params non-zero elements
    # Must match base_layer dtype and device
    return mask
```

## Scaling Mechanism

SHiRA supports dynamic scaling at inference:

```python
# During training: scaling = 1.0
layer.scaling["adapter"] = 1.0

# At inference: adjust adaptation magnitude
layer.set_scale("adapter", 0.5)   # Reduce adaptation
layer.set_scale("adapter", 1.5)   # Increase adaptation
layer.set_scale("adapter", 0.0)   # Disable (equivalent to disable_adapters)
```

**Use Cases**:
- Hyperparameter search without retraining
- Ensemble with different scales
- Gradual adaptation strength adjustment

## Design Patterns

### 1. Sparse COO Tensor Storage
Uses PyTorch's sparse COO (Coordinate format) tensors:
- Efficient storage for sparse matrices
- GPU operations supported
- Easy conversion to dense when needed

### 2. Mask-Driven Architecture
Mask determines structure, weights are learned:
```python
mask → indices (fixed) → shira_indices[adapter]
values (learnable) → shira_weight[adapter]
```

### 3. Deep Copy for Forward Pass
```python
new_weight = copy.deepcopy(base_layer.weight.data)
new_weight += delta_weight
```
Prevents accidental modification of base weights

### 4. Validation-Heavy Initialization
Extensive checks during `update_layer`:
- Parameter budget validation
- Mask dimension validation
- Device and dtype consistency

## Multi-GPU Considerations

**Device Synchronization**:
```python
# In get_delta_weight
self.shira_indices[adapter] = self.shira_indices[adapter].to(self.shira_weight[adapter].device)
```

**Why Needed**: In multi-GPU environments, indices may be on wrong device initially. This ensures indices match weight device before sparse tensor construction.

## Initialization Strategies

### Zero Initialization (Default, init_weights=True)
```python
shira_weight = torch.zeros(num_shira_params)
```
- Start with no adaptation
- Gradual learning of sparse structure
- Recommended for training

### Random Initialization (init_weights=False)
```python
shira_weight = torch.randn(num_shira_params)
```
- Used for testing/debugging
- Non-zero initial adaptation
- Not recommended for production

## Sparse Tensor Properties

**Memory Efficiency**:
```python
Dense: m × n × sizeof(float32)
Sparse: 2 × num_nonzero × sizeof(int32) + num_nonzero × sizeof(float32)

Example (4096×4096, r=32):
Dense: 16,777,216 × 4 = 67 MB
Sparse: 2 × 262,144 × 4 + 262,144 × 4 = 3.1 MB
```

**Computation**:
- Sparse matrix operations when possible
- Densification may occur during certain ops
- PyTorch handles optimization

## Limitations

1. **Layer Support**: Only nn.Linear currently supported
2. **Nested Layers**: Does not support nested base layers (raises ValueError)
3. **Merge Overhead**: Deep copy adds memory overhead during forward pass
4. **Sparse Ops**: Not all operations optimized for sparse tensors

## Integration Points

- Imports `BaseTunerLayer` from `peft.tuners.tuners_utils`
- Uses `check_adapters_to_merge` for merge validation
- PyTorch sparse tensor functionality
- Deep copy from Python's copy module

## Usage Example

```python
import torch.nn as nn
from peft.tuners.shira.layer import Linear
from peft.tuners.shira.mask_functions import random_mask

# Create base layer
base_layer = nn.Linear(768, 768)

# Generate mask
mask = random_mask(base_layer, r=32)

# Create SHiRA layer
shira_layer = Linear(
    base_layer=base_layer,
    mask=mask,
    adapter_name="default",
    r=32,
    fan_in_fan_out=False,
    init_weights=True
)

# Forward pass
output = shira_layer(input_tensor)

# Adjust scaling at inference
shira_layer.set_scale("default", 0.8)

# Merge for faster inference
shira_layer.merge(safe_merge=True)
```

## Advanced Usage: Custom Masks

```python
def structured_mask(base_layer, r, block_size=16):
    """Create block-structured sparse pattern"""
    m, n = base_layer.weight.shape
    num_params = r * (m + n)

    mask = torch.zeros(m, n, dtype=base_layer.weight.dtype, device=base_layer.weight.device)

    # Select random blocks
    num_blocks = num_params // (block_size * block_size)
    blocks_m = m // block_size
    blocks_n = n // block_size

    selected_blocks = torch.randperm(blocks_m * blocks_n)[:num_blocks]

    for block_idx in selected_blocks:
        block_row = (block_idx // blocks_n) * block_size
        block_col = (block_idx % blocks_n) * block_size
        mask[block_row:block_row+block_size, block_col:block_col+block_size] = 1.0

    return mask

# Use custom mask
config = ShiraConfig(r=32)
config.mask_fn = structured_mask
model = get_peft_model(base_model, config)
```

## Implementation Notes

1. **Sparse Tensor Device**: Automatically handles device transfers in get_delta_weight
2. **Scaling Flexibility**: Can be adjusted without retraining
3. **Merge Performance**: Safe merge checks for NaN values
4. **Deep Copy Usage**: Necessary to avoid modifying frozen base weights
5. **Index Storage**: Uses int32 by default (via .to(torch.int))

## References

- **Concept**: Sparse high-rank adaptation as alternative to low-rank dense adaptation
- **Parameter Budget**: Matches LoRA parameter count while achieving higher effective rank
- **Sparsity**: Enables high-rank approximations with same memory footprint as low-rank
