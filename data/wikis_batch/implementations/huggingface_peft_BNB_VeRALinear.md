# Implementation: BitsAndBytes Quantized VeRA Linear Layers

## File Location
`src/peft/tuners/vera/bnb.py`

## Overview
This module implements VeRA (Vector-based Random Matrix Adaptation) for BitsAndBytes quantized linear layers. VeRA uses shared random projection matrices (similar to RandLoRA) but with simpler per-layer trainable scaling vectors, providing an even more parameter-efficient alternative to LoRA.

## Key Components

### Class: `Linear8bitLt`
VeRA implementation for 8-bit quantized linear layers.

**Conditional Definition:**
- Only defined if BitsAndBytes is available (`is_bnb_available()`)

**Inheritance:**
- `torch.nn.Module`
- `VeraLayer`

**Key Features:**
- Shared random projection matrices (vera_A, vera_B) across layers
- Per-layer trainable scaling vectors (lambda_d, lambda_b)
- Supports merge/unmerge operations
- Works with 8-bit quantization
- Simpler than RandLoRA (vectors vs matrices)

**Constructor Parameters:**
- `base_layer` (torch.nn.Module): BitsAndBytes 8-bit quantized layer
- `adapter_name` (str): Adapter name
- `vera_A`: Shared random projection A (r × max_features)
- `vera_B`: Shared random projection B (max_features × r)
- `r` (int): Rank of adaptation (default: 0)
- `vera_dropout` (float): Dropout probability (default: 0.0)
- `fan_in_fan_out` (bool): Weight transpose flag (default: False)
- `init_weights` (bool): Initialize scaling vectors (default: True)
- `d_initial` (float): Initial scaling value (default: 0.1)

**Key Attributes:**
- `fan_in_fan_out`: Determines weight orientation
- `d_initial`: Controls initialization magnitude

### Method: `merge(safe_merge, adapter_names)` (Linear8bitLt)
Merges VeRA adapters into 8-bit quantized base weights.

**Warning:**
"Merge vera module to 8-bit linear may get different generations due to rounding errors."

**Process:**
1. Warn about potential duplicate merges
2. Validate adapters to merge
3. For each adapter:
   - Compute delta weight from get_delta_weight
   - Dequantize base weights
   - Add delta to dequantized weights
   - Check for NaNs if safe_merge enabled
   - Re-quantize as Int8Params
   - Reset gradients
   - Track merged adapter

### Method: `unmerge()` (Linear8bitLt)
Reverses merge operation for 8-bit layers.

**Process:**
1. Check if adapters merged
2. For each merged adapter (LIFO):
   - Compute delta weight
   - Dequantize current weights
   - Subtract delta
   - Re-quantize as Int8Params
   - Reset gradients

### Method: `get_delta_weight(adapter)` (Linear8bitLt)
Computes the delta weight contribution from a VeRA adapter.

**VeRA-Specific Computation:**
```python
# Retrieve shared projections
vera_A = self.vera_A[adapter]  # (r, max_features)
vera_B = self.vera_B[adapter]  # (max_features, r)

# Retrieve per-layer scaling vectors
lambda_d = self.vera_lambda_d[adapter]  # (r,)
lambda_b = self.vera_lambda_b[adapter]  # (out_features,)

# Slice to layer dimensions
sliced_A = vera_A[:, :self.in_features]
sliced_B = vera_B[:self.out_features, :]

# Apply scaling and compute delta
lambda_b = lambda_b.unsqueeze(-1)  # (out_features, 1)
lambda_d = lambda_d.unsqueeze(-1)  # (r, 1)
output = (lambda_b * sliced_B) @ (lambda_d * sliced_A)
output = transpose(output, self.fan_in_fan_out)
```

**Key Points:**
- Element-wise multiplication with scaling vectors
- Lambda_b scales output dimension
- Lambda_d scales latent dimension
- Transpose applied based on fan_in_fan_out

### Method: `forward(x: torch.Tensor, *args, **kwargs)` (Linear8bitLt)
Forward pass with VeRA adaptation.

**Implementation Flow:**
1. Handle disable_adapters and merged states
2. Execute base layer
3. For each active adapter:
   - Handle dtype conversion
   - Retrieve lambda_d, lambda_b, vera_A, vera_B
   - Slice projections to layer dimensions
   - Apply dropout
   - Compute: `lambda_b * linear(lambda_d * linear(dropout(x), sliced_A), sliced_B)`
   - Add to result

**Efficient Computation:**
```python
x_temp = dropout(x.to(lambda_d.dtype))
adapter_output = lambda_b * torch.nn.functional.linear(
    lambda_d * torch.nn.functional.linear(x_temp, sliced_A),
    sliced_B
)
result = result + adapter_output
```

**Dtype Preservation:**
Ensures output matches input dtype: `return result.to(x.dtype)`

### Class: `Linear4bit`
VeRA implementation for 4-bit quantized linear layers.

**Conditional Definition:**
- Only defined if BitsAndBytes 4-bit is available (`is_bnb_4bit_available()`)

**Key Features:**
- Same VeRA approach as 8-bit
- Works with 4-bit quantization
- Includes defensive cloning
- More memory efficient

**Constructor:**
Identical to `Linear8bitLt` with same parameters.

### Method: `merge(safe_merge, adapter_names)` (Linear4bit)
Merges VeRA adapters into 4-bit quantized weights.

**Process:**
Similar to 8-bit with 4-bit specific operations:
1. Dequantize with `bnb.functional.dequantize_4bit`
2. Add delta weight
3. Clean up kwargs (remove '_' prefixed attributes)
4. Re-quantize as Params4bit

### Method: `forward(x: torch.Tensor, *args, **kwargs)` (Linear4bit)
Forward pass for 4-bit with VeRA.

**Key Difference:**
Includes defensive cloning:
```python
result = result.clone()
```

**Otherwise identical to 8-bit version.**

## Dependencies
- `torch`
- `warnings`
- `bitsandbytes as bnb`
- `peft.import_utils.{is_bnb_available, is_bnb_4bit_available}`
- `peft.tuners.tuners_utils.check_adapters_to_merge`
- `peft.utils.integrations.dequantize_bnb_weight`
- `peft.utils.other.transpose`
- `peft.tuners.vera.layer.VeraLayer`

## Key Characteristics

### VeRA Innovation

**Shared Random Projections:**
- vera_A and vera_B shared across ALL layers
- Initialized once with random values
- Frozen during training (not updated)

**Per-Layer Scaling Vectors:**
- `vera_lambda_d`: Scales latent dimension (rank r)
- `vera_lambda_b`: Scales output dimension (out_features)
- Only these vectors are trainable

**Comparison to RandLoRA:**
- **RandLoRA**: Per-layer lambda and gamma matrices
- **VeRA**: Per-layer lambda vectors (even simpler)
- VeRA has fewer parameters than RandLoRA

### Parameter Efficiency

**Parameter Count:**
- **LoRA per layer**: `r * in_features + r * out_features`
- **VeRA per layer**: `r + out_features`
- Dramatic reduction for large r or large layers

**Shared Resources:**
- vera_A: `(r, max_in_features)` shared
- vera_B: `(max_out_features, r)` shared
- Total trainable per layer: `r + out_features` scalars

### Dimension Handling

**Adaptive Slicing:**
Projections initialized with max dimensions:
```python
sliced_A = vera_A[:, :self.in_features]
sliced_B = vera_B[:self.out_features, :]
```

Each layer uses only needed portion.

**Broadcasting:**
Scaling vectors broadcast to match dimensions:
```python
lambda_b = lambda_b.unsqueeze(-1)  # (out_features, 1)
lambda_d = lambda_d.unsqueeze(-1)  # (r, 1)
```

### Merge/Unmerge Support

**Delta Weight Computation:**
Full delta computed from shared projections and scaling:
```
ΔW = (lambda_b ⊙ B) @ (lambda_d ⊙ A)
```

**Merge Process:**
1. Dequantize base weights
2. Add delta: `W' = W + ΔW`
3. Re-quantize

**Unmerge Process:**
1. Dequantize merged weights
2. Subtract delta: `W = W' - ΔW`
3. Re-quantize

## Usage Context
These classes are used when:
1. BitsAndBytes quantization applied
2. VeRA configuration specified with shared projections
3. Need maximum parameter efficiency

Particularly effective for:
- Extremely deep networks
- Multiple adapter training
- Ultra-low parameter budgets
- Resource-constrained environments

## Notes
- Both classes use `__repr__` prefix "vera."
- Merge warnings about potential rounding errors
- Warnings about duplicate merges (can merge multiple times)
- CPU bf16/fp16 cast to fp32 for matmul performance
- Defensive clone for 4-bit forward pass
- Simpler than RandLoRA but still very effective
- Initial scaling (d_initial) controls adaptation magnitude
- fan_in_fan_out handled consistently
- Detailed docstrings explain VeRA-specific computation
- Dtype preservation ensures stability: `result.to(x.dtype)`
