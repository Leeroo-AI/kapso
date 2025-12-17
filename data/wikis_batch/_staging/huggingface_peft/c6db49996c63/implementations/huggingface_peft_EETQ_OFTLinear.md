# Implementation: EETQ OFT Linear Layer

## File Location
`src/peft/tuners/oft/eetq.py`

## Overview
This module implements OFT (Orthogonal Fine-Tuning) for EETQ (Easy and Efficient Quantization Toolkit) quantized linear layers. It applies orthogonal transformations to inputs of EETQ's int8 quantized layers, with explicit blocks on merge/unmerge operations.

## Key Components

### Class: `EetqOFTLinear`
Applies OFT adaptation to EETQ quantized linear layers.

**Conditional Definition:**
- Only defined if EETQ is available (`is_eetq_available()`)

**Inheritance:**
- `torch.nn.Module`
- `OFTLayer`

**Key Features:**
- Orthogonal transformation of inputs
- Compatible with EETQ int8 quantization
- Block-diagonal structure for efficiency
- Explicitly disables merge/unmerge
- Supports Cayley-Neumann parameterization
- COFT variant available

**Constructor Parameters:**
- `base_layer`: EETQ quantized base layer
- `adapter_name` (str): Adapter name
- `r` (int): Rank parameter (default: 0)
- `oft_block_size` (int): Orthogonal block size (default: 0)
- `module_dropout` (float): Dropout probability (default: 0.0)
- `init_weights` (bool): Initialize weights (default: True)
- `coft` (bool): Use constrained OFT (default: False)
- `eps` (float): Numerical stability (default: 6e-5)
- `block_share` (bool): Share blocks (default: False)
- `use_cayley_neumann` (bool): Cayley parameterization (default: False)
- `num_cayley_neumann_terms` (int): Neumann terms (default: 5)
- `fan_in_fan_out` (bool): Weight transpose (default: False)

**Attributes:**
- `quant_linear_module`: Reference to EETQ base layer

### Method: `forward(x: torch.Tensor)`
Forward pass with OFT adaptation on EETQ quantized weights.

**Implementation Flow:**
1. If adapters disabled, return quantized base output
2. For each active adapter:
   - Retrieve oft_R orthogonal transformation
   - Handle dtype conversion when not in autocast
   - Apply transformation: `x = oft_R(x)`
3. Pass transformed input through EETQ quantized layer
4. Convert result to expected dtype if needed
5. Return result

**Transformation Pattern:**
```python
for active_adapter in self.active_adapters:
    oft_R = self.oft_R[active_adapter]

    requires_conversion = not torch.is_autocast_enabled()
    if requires_conversion:
        expected_dtype = x.dtype
        x = self._cast_input_dtype(x, oft_R.weight.dtype)

    x = oft_R(x)

result = self.quant_linear_module(x)
if requires_conversion:
    result = result.to(expected_dtype)
```

### Method: `merge(safe_merge, adapter_names)`
**Explicitly Not Supported** - Raises `AttributeError`

**Error Message:**
"Merging LoRA layers is not supported for Eetq layers."

**Note:** Message says "LoRA" but this is OFT - likely copy-paste from EETQ LoRA implementation.

### Method: `unmerge()`
**Explicitly Not Supported** - Raises `AttributeError`

**Error Message:**
"Unmerging LoRA layers is not supported for Eetq layers."

**Same Note:** Message references LoRA instead of OFT.

### Function: `dispatch_eetq(target, adapter_name, **kwargs)`
Factory function to create EETQ OFT adapter.

**Parameters:**
- `target` (torch.nn.Module): Target layer
- `adapter_name` (str): Adapter name
- `**kwargs`: Configuration parameters

**Returns:**
- New `EetqOFTLinear` instance if target is EETQ quantized, None otherwise

**Logic:**
1. Extracts base layer if target is tuner layer
2. Checks if EETQ available
3. Verifies target is `EetqLinear`
4. Creates wrapper if conditions met
5. Sets `weight` attribute to base layer weight
6. Sets `bias` if present in base layer

## Dependencies
- `torch`
- `typing.{Any, Optional}`
- `peft.import_utils.is_eetq_available`
- `peft.tuners.oft.layer.OFTLayer`
- `peft.tuners.tuners_utils.BaseTunerLayer`
- `eetq.EetqLinear` (conditional import)

## Key Characteristics

### EETQ Integration

**Int8 Quantization:**
- Works with EETQ's int8 weight quantization
- Transformations applied to activations (not weights)
- Preserves quantization benefits

**No Merge/Unmerge:**
Similar to EETQ LoRA implementation:
- EETQ format doesn't support weight modification
- Safer to error than produce incorrect results
- Forward-only adaptation

**Weight Handling:**
- Directly references `weight` attribute
- Also handles `bias` if present
- No separate `qweight` needed

### OFT Characteristics

**Orthogonal Input Transformation:**
```
output = EETQ_layer(OFT(input))
```

**Block-Diagonal Structure:**
- Controlled by `oft_block_size`
- Each block is independent orthogonal matrix
- Reduces parameters and computation

**Parameterization Options:**
- **Standard**: Direct orthogonal matrices
- **Cayley-Neumann**: Ensures orthogonality via Cayley transform
- **COFT**: Constrained variant for stability

### Limitations

**Explicit Restrictions:**
1. **No Merge**: Cannot merge OFT into quantized weights
2. **No Unmerge**: Cannot reverse merge (since merge not supported)
3. **Forward Only**: Adapters only applied during forward pass

**Design Rationale:**
- EETQ's quantization format incompatible with weight modification
- Input transformation doesn't require weight access
- Maintains separation between quantization and adaptation

### Dtype Management

**Conversion Logic:**
- Checks autocast status
- Converts input to oft_R weight dtype
- Applies transformation
- Ensures result matches expected dtype

**Careful Handling:**
Multiple conversions ensure:
- Compatibility with EETQ int8 operations
- Proper OFT transformation
- Correct output dtype

## Usage Context
This adapter is dispatched when:
1. EETQ library installed
2. Target is `EetqLinear` layer
3. OFT configuration applied

Useful for:
- Stable adaptation of EETQ models
- Property-preserving fine-tuning
- Scenarios where orthogonality is beneficial

## Notes
- The `__repr__` method prefixes output with "oft."
- Error messages incorrectly reference "LoRA" (should be "OFT")
- Conditional class definition prevents import errors
- Bias handling is optional (checked with hasattr)
- Dual naming (`base_layer` and `quant_linear_module`) maintains consistency
- Default `oft_block_size=0` may need explicit setting
- fan_in_fan_out parameter supports different weight layouts
- Input transformation approach avoids need for weight access
- Clear separation between quantization and adaptation layers
