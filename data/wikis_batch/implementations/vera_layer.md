# VeRA Layer Implementation

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/vera/layer.py`
- **Lines**: 291
- **Purpose**: VeRA (Vector-based Random Matrix Adaptation) adapter layer implementation

## Overview

This module implements VeRA adapter layers, a parameter-efficient fine-tuning method that uses shared random projection matrices with learned scaling vectors. VeRA reduces trainable parameters compared to LoRA by using shared frozen random matrices (vera_A and vera_B) combined with trainable diagonal scaling vectors (lambda_b and lambda_d).

## Key Components

### VeraLayer (Base Class)

**Purpose**: Base layer for VeRA adapters that manages adapter parameters and shared projection matrices.

**Attributes**:
- `adapter_layer_names`: `("vera_lambda_b", "vera_lambda_d")` - Trainable scaling parameters
- `other_param_names`: `("vera_A", "vera_B")` - Shared frozen projection matrices
- `vera_lambda_b`: ParameterDict storing output scaling vectors (shape: out_features)
- `vera_lambda_d`: ParameterDict storing rank scaling vectors (shape: r)
- `vera_A`: BufferDict storing shared input projection matrix (shape: r × max_in_features)
- `vera_B`: BufferDict storing shared output projection matrix (shape: max_out_features × r)
- `r`: Dict mapping adapter names to their ranks
- `vera_dropout`: ModuleDict for dropout layers per adapter

**Key Methods**:

1. **`__init__(base_layer, **kwargs)`**
   - Initializes base layer wrapper
   - Extracts in_features and out_features from base layer (Linear or Conv1D)
   - Sets up adapter parameter dictionaries

2. **`update_layer(adapter_name, vera_A, vera_B, r, vera_dropout, init_weights, d_initial=0.1, inference_mode=False)`**
   - Adds or updates a VeRA adapter
   - Parameters:
     - `r`: Rank of the adapter (must be positive)
     - `vera_A`, `vera_B`: References to shared BufferDict
     - `vera_dropout`: Dropout probability
     - `d_initial`: Initial value for lambda_d (default: 0.1)
   - Creates trainable parameters:
     - `vera_lambda_b[adapter_name]`: Initialized to ones (shape: out_features)
     - `vera_lambda_d[adapter_name]`: Initialized to d_initial (shape: r)
   - Handles multiple adapters by reusing existing vera_A/vera_B buffers
   - Validates buffer dimensions for compatibility

3. **`reset_vera_parameters(adapter_name, d_initial=0.1)`**
   - Initializes lambda_d to d_initial
   - Initializes lambda_b to zeros

### Linear (Implementation Class)

**Purpose**: VeRA implementation for Linear layers (inherits from nn.Linear and VeraLayer).

**Constructor Parameters**:
- `base_layer`: Original Linear layer to wrap
- `vera_A`, `vera_B`: Shared projection matrices (BufferDict)
- `adapter_name`: Name of the adapter
- `r`: Rank dimension
- `vera_dropout`: Dropout probability
- `fan_in_fan_out`: Whether layer stores weights as (fan_in, fan_out)
- `is_target_conv_1d_layer`: Flag for Conv1D layers
- `init_weights`: Whether to initialize parameters
- `d_initial`: Initial value for lambda_d scaling

**Key Methods**:

1. **`merge(safe_merge=False, adapter_names=None)`**
   - Merges active adapter weights into base weights
   - Formula: `weight = weight + lambda_b * vera_B @ (lambda_d * vera_A)`
   - Parameters:
     - `safe_merge`: If True, checks for NaNs before committing merge
     - `adapter_names`: List of adapters to merge (None = all active)
   - Updates `merged_adapters` list

2. **`unmerge()`**
   - Removes merged adapter weights from base weights
   - Reverses the merge operation
   - Pops adapters from `merged_adapters` list

3. **`get_delta_weight(adapter)`**
   - Computes delta weight for an adapter
   - Algorithm:
     1. Slice vera_A to match in_features: `sliced_A = vera_A[:, :in_features]`
     2. Slice vera_B to match out_features: `sliced_B = vera_B[:out_features, :]`
     3. Apply scaling: `delta = (lambda_b * sliced_B) @ (lambda_d * sliced_A)`
     4. Handle fan_in_fan_out transpose if needed
   - CPU float16/bfloat16 handling: Casts to float32 for computation

4. **`forward(x, *args, **kwargs)`**
   - Forward pass with VeRA adaptation
   - Logic flow:
     ```
     if disable_adapters:
         if merged: unmerge()
         return base_layer(x)
     elif merged:
         return base_layer(x)  # Already includes adapter
     else:
         result = base_layer(x)
         for each active adapter:
             # Slice shared matrices
             sliced_A = vera_A[:, :in_features]
             sliced_B = vera_B[:out_features, :]
             # Apply VeRA transformation
             result += lambda_b * F.linear(lambda_d * F.linear(dropout(x), sliced_A), sliced_B)
         return result
     ```
   - Preserves input dtype throughout computation

## Mathematical Formulation

VeRA adaptation follows:

```
output = base_layer(x) + lambda_b ⊙ (B @ (lambda_d ⊙ (A @ x)))
```

Where:
- `A`: Shared frozen projection matrix (r × d_in)
- `B`: Shared frozen projection matrix (d_out × r)
- `lambda_d`: Learned per-layer rank scaling (r,)
- `lambda_b`: Learned per-layer output scaling (d_out,)
- `⊙`: Element-wise multiplication

## Design Patterns

### 1. Shared Matrix Slicing
VeRA initializes shared matrices with maximum dimensions across all adapted layers. During forward pass, required submatrices are sliced:
```python
sliced_A = vera_A[:, : self.in_features]
sliced_B = vera_B[: self.out_features, :]
```

### 2. Multi-Adapter Management
- First adapter initializes vera_A and vera_B buffers
- Subsequent adapters reuse existing buffers
- Dimension validation ensures compatibility
- Each adapter has independent lambda_b and lambda_d parameters

### 3. CPU Float16 Handling
When merging on CPU with float16/bfloat16:
```python
cast_to_fp32 = device.type == "cpu" and (dtype in [torch.float16, torch.bfloat16])
if cast_to_fp32:
    # Cast to float32 for computation
    # Cast back to original dtype
```

## Parameter Efficiency

For a layer with dimensions d_in × d_out:
- **LoRA parameters**: r(d_in + d_out)
- **VeRA parameters**: r + d_out (lambda_d + lambda_b)
- **Shared parameters**: r(d_in + d_out) (vera_A + vera_B, frozen and shared)

VeRA dramatically reduces trainable parameters while maintaining adaptation capability.

## Integration Points

- Imports `BaseTunerLayer` from `peft.tuners.tuners_utils`
- Uses `check_adapters_to_merge` for merge validation
- Supports Conv1D layers from transformers.pytorch_utils
- Uses `BufferDict` from `peft.tuners._buffer_dict` for shared matrices

## Usage Example

```python
# Initialize VeRA layer
vera_layer = Linear(
    base_layer=nn.Linear(768, 768),
    vera_A=vera_A_buffer,
    vera_B=vera_B_buffer,
    adapter_name="default",
    r=256,
    vera_dropout=0.1,
    d_initial=0.1
)

# Forward pass
output = vera_layer(input_tensor)

# Merge for inference
vera_layer.merge(safe_merge=True)
```

## Implementation Notes

1. **Initialization**: Lambda_d initialized to small positive value (0.1), lambda_b to zeros
2. **Dropout**: Applied before first projection (on input x)
3. **Device Management**: Ensures lambda and matrix tensors are on same device
4. **Inference Mode**: Supports disabling adapters for base model evaluation
5. **Merge Safety**: Optional NaN checking during merge operations

## References

- Paper: https://huggingface.co/papers/2310.11454
- Based on the principle of shared random projections with learned scaling
