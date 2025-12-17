# Implementation: BitsAndBytes Quantized RandLoRA Linear Layers

## File Location
`src/peft/tuners/randlora/bnb.py`

## Overview
This module implements RandLoRA (Random Low-Rank Adaptation) for BitsAndBytes quantized linear layers. RandLoRA uses shared random projection matrices across layers with per-layer trainable scaling parameters, providing even greater parameter efficiency than standard LoRA.

## Key Components

### Class: `Linear8bitLt`
RandLoRA implementation for 8-bit quantized linear layers.

**Conditional Definition:**
- Only defined if BitsAndBytes is available (`is_bnb_available()`)

**Inheritance:**
- `torch.nn.Module`
- `RandLoraLayer`

**Key Features:**
- Shared random bases (randlora_A, randlora_B) across layers
- Per-layer trainable parameters (lambda, gamma)
- Supports merge/unmerge operations
- Works with 8-bit quantization
- Uses custom gradient computation with UniqueBaseGrad

**Constructor Parameters:**
- `base_layer` (torch.nn.Module): BitsAndBytes 8-bit quantized layer
- `adapter_name` (str): Adapter name
- `randlora_A`: Shared random projection A (small dimension)
- `randlora_B`: Shared random projection B (large dimension)
- `r` (int): Rank (default: 0)
- `randlora_alpha` (int): Scaling factor (default: 0)
- `randlora_dropout` (float): Dropout probability (default: 0.0)
- `fan_in_fan_out` (bool): Weight transpose flag (default: False)
- `init_weights` (bool): Initialize trainable weights (default: True)

**Key Attributes:**
- `fan_in_fan_out`: Determines weight orientation

### Method: `merge(safe_merge, adapter_names)` (Linear8bitLt)
Merges RandLoRA adapters into 8-bit quantized base weights.

**Warning:**
"Merge RandLora module to 8-bit linear may get different generations due to rounding errors."

**Process:**
1. Validate adapters to merge
2. For each adapter:
   - Compute delta weight from get_delta_weight
   - Dequantize base weights
   - Add delta to dequantized weights
   - Check for NaNs if safe_merge enabled
   - Re-quantize merged weights as Int8Params
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

### Method: `get_scaled_bases(adapter, device)` (Linear8bitLt)
Retrieves and scales the shared random bases for a specific adapter.

**Core RandLoRA Logic:**
1. Retrieve shared randlora_A and randlora_B
2. Retrieve per-layer lambda and gamma parameters
3. Handle CPU bf16/fp16 casting to fp32
4. Determine min_dim and max_dim for layer
5. Slice appropriate submatrices from shared bases
6. Apply UniqueBaseGrad for trainable scaling: `UniqueBaseGrad.apply(sliced_A, lambda, gamma)`
7. Flatten over rank and basis dimensions
8. Return in correct order for layer dimensions

**Dimension Handling:**
```python
min_dim, max_dim = min(out_features, in_features), max(out_features, in_features)
sliced_A = randlora_A[:, :num_bases, :min_dim]
sliced_B = randlora_B[:max_dim, :num_bases, :]
```

**UniqueBaseGrad:**
Custom autograd function applies lambda and gamma scaling while ensuring proper gradient flow to shared bases.

### Method: `get_delta_weight(adapter)` (Linear8bitLt)
Computes the full delta weight for merging.

**Process:**
1. Get scaled bases from get_scaled_bases
2. Compute update: `update_B @ update_A`
3. Apply transpose if fan_in_fan_out
4. Multiply by scaling factor
5. Return delta tensor

### Method: `forward(x: torch.Tensor, *args, **kwargs)` (Linear8bitLt)
Forward pass with RandLoRA adaptation.

**Implementation Flow:**
1. Handle disable_adapters and merged states
2. For each active adapter:
   - Get scaled bases for adapter
   - Handle dtype conversion
   - Apply dropout
   - Compute: `linear(linear(dropout(x), update_B), update_A)`
   - Convert to expected dtype
   - Scale and add to result

**Efficient Computation:**
Uses functional linear operations for adapter computation:
```python
adapter_output = torch.nn.functional.linear(
    torch.nn.functional.linear(dropout(x), update_B),
    update_A
)
result = result + adapter_output * scaling
```

### Class: `Linear4bit`
RandLoRA implementation for 4-bit quantized linear layers.

**Conditional Definition:**
- Only defined if BitsAndBytes 4-bit is available (`is_bnb_4bit_available()`)

**Key Features:**
- Same RandLoRA approach as 8-bit
- Works with 4-bit quantization (Params4bit)
- More memory efficient
- Includes defensive cloning

**Methods:**
Nearly identical to Linear8bitLt with differences:
- Uses `bnb.functional.dequantize_4bit` and `bnb.nn.Params4bit`
- Includes `result.clone()` for 4-bit backward compatibility
- Different weight attribute handling

## Helper Class: `UniqueBaseGrad`

While not defined in this file, it's a critical component:
- Custom `torch.autograd.Function`
- Applies lambda and gamma scaling to shared bases
- Ensures gradients flow correctly to per-layer parameters
- Prevents gradient interference between layers sharing bases

## Dependencies
- `torch`
- `warnings`
- `bitsandbytes as bnb`
- `peft.import_utils.{is_bnb_available, is_bnb_4bit_available}`
- `peft.tuners.tuners_utils.check_adapters_to_merge`
- `peft.utils.integrations.dequantize_bnb_weight`
- `peft.utils.other.transpose`
- `peft.tuners.randlora.layer.{RandLoraLayer, UniqueBaseGrad}`

## Key Characteristics

### RandLoRA Innovation

**Shared Random Bases:**
- randlora_A and randlora_B shared across ALL layers
- Initialized once, frozen during training
- Significantly reduces memory footprint

**Per-Layer Trainables:**
- `randlora_lambda`: Per-layer scaling for small basis
- `randlora_gamma`: Per-layer scaling for large basis
- Only these parameters trained

**Memory Efficiency:**
- Standard LoRA: Each layer has own A and B matrices
- RandLoRA: Shared A and B, only small lambda/gamma per layer
- Massive parameter reduction for deep networks

### Gradient Flow
**UniqueBaseGrad Function:**
Essential for proper training:
- Shared bases are frozen
- Gradients flow to lambda and gamma
- Prevents gradient conflicts across layers
- Enables end-to-end training

### Dimension Handling
**Adaptive Slicing:**
- Bases initialized with max dimensions across all layers
- Each layer slices needed submatrices
- Handles varying layer sizes efficiently

**Orientation:**
- fan_in_fan_out determines transpose behavior
- Ensures compatibility with different layer types

### Merge/Unmerge Support
Unlike most quantized adapters, RandLoRA supports merging:
- Dequantizes weights
- Computes full delta from shared bases
- Re-quantizes merged result
- Warns about potential rounding errors

## Usage Context
These classes are used when:
1. BitsAndBytes quantization applied
2. RandLoRA configuration specified
3. Shared bases provided in configuration
4. Need extreme parameter efficiency

Particularly effective for:
- Very deep networks
- Multiple task adaptation
- Memory-constrained deployment

## Notes
- Both classes use `__repr__` prefix "randlora."
- Merge warnings about rounding errors important for generation tasks
- Flattening over rank/basis dimensions improves memory efficiency
- CPU bf16/fp16 cast to fp32 for matmul performance
- Defensive clone for 4-bit referenced from adalora.py and lora.py
- Small epsilon (1e-5) would be in ranknum if used (not present here)
- Complex but extremely parameter-efficient design
- Requires careful initialization of shared bases
