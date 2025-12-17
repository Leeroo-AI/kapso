# Implementation: AWQ OFT Linear Layer

## File Location
`src/peft/tuners/oft/awq.py`

## Overview
This module implements OFT (Orthogonal Fine-Tuning) for AWQ (Activation-aware Weight Quantization) quantized linear layers. It combines OFT's orthogonal transformations with AWQ's quantization, requiring AutoAWQ version 0.2.0 or higher.

## Key Components

### Class: `AwqOFTLinear`
Applies OFT adaptation to AWQ quantized linear layers.

**Inheritance:**
- `torch.nn.Module`
- `OFTLayer`

**Key Features:**
- Orthogonal transformation of layer inputs
- Compatible with AWQ quantization (version 0.2.0+)
- Supports block-diagonal structure
- Optional Cayley-Neumann parameterization
- COFT variant support
- Block sharing capability

**Constructor Parameters:**
- `base_layer`: AWQ quantized base layer
- `adapter_name` (str): Adapter name
- `r` (int): Rank parameter (default: 0)
- `oft_block_size` (int): Size of orthogonal blocks (default: 32)
- `module_dropout` (float): Dropout probability (default: 0.0)
- `coft` (bool): Use constrained OFT (default: False)
- `eps` (float): Numerical stability epsilon (default: 6e-5)
- `block_share` (bool): Share blocks (default: False)
- `fan_in_fan_out` (bool): Transpose representation (default: False)
- `init_weights` (bool): Initialize weights (default: True)
- `use_cayley_neumann` (bool): Cayley-Neumann parameterization (default: False)
- `num_cayley_neumann_terms` (int): Neumann series terms (default: 5)

**Attributes:**
- `quant_linear_module`: Reference to base AWQ layer (backward compatibility)

### Method: `forward(x: torch.Tensor)`
Forward pass with OFT adaptation on AWQ quantized weights.

**Implementation Flow:**
1. If adapters disabled, return base quantized output
2. For each active adapter:
   - Retrieve oft_R orthogonal transformation
   - Handle dtype conversion when not in autocast
   - Apply transformation to input: `x = oft_R(x)`
   - Convert back to expected dtype
3. Pass transformed input through AWQ quantized base layer
4. Return result

**Key Pattern:**
```python
for active_adapter in self.active_adapters:
    oft_R = self.oft_R[active_adapter]

    requires_conversion = not torch.is_autocast_enabled()
    if requires_conversion:
        expected_dtype = x.dtype
        x = self._cast_input_dtype(x, oft_R.weight.dtype)

    x = oft_R(x)

    if requires_conversion:
        x = x.to(expected_dtype)

result = self.quant_linear_module(x)
```

**Transformation Application:**
OFT transforms input before quantized computation (not additive like LoRA).

### Function: `dispatch_awq(target, adapter_name, **kwargs)`
Factory function with version validation for AWQ OFT adapter.

**Parameters:**
- `target` (torch.nn.Module): Target layer
- `adapter_name` (str): Adapter name
- `**kwargs`: Configuration parameters

**Returns:**
- New `AwqOFTLinear` instance if conditions met, None otherwise

**Version Validation:**
1. Checks if AutoAWQ is available
2. Verifies target is `WQLinear_GEMM`
3. **Validates AutoAWQ version >= 0.2.0**
4. Raises `ImportError` if version incompatible

**Version Check Logic:**
```python
AUTOAWQ_MINIMUM_VERSION = packaging.version.parse("0.2.0")
version_autoawq = packaging.version.parse(importlib_metadata.version("autoawq"))

if AUTOAWQ_MINIMUM_VERSION > version_autoawq:
    raise ImportError(
        f"Found an incompatible version of auto-awq. Found version {version_autoawq}, "
        f"but only versions above {AUTOAWQ_MINIMUM_VERSION} are supported for PEFT."
    )
```

**Setup:**
- Creates wrapper if version check passes
- Sets `qweight` attribute to base layer's quantized weights

## Dependencies
- `torch`
- `packaging.version` - Version comparison
- `importlib.metadata` - Package version checking
- `typing.{Any, Optional}`
- `peft.import_utils.is_auto_awq_available`
- `peft.tuners.oft.layer.OFTLayer`
- `peft.tuners.tuners_utils.BaseTunerLayer`
- `awq.modules.linear.WQLinear_GEMM` (conditional import)

## Key Characteristics

### OFT with AWQ

**Orthogonal Input Transformation:**
- Applies orthogonal matrix to input activations
- Preserves norms and angles
- Maintains AWQ quantization benefits
- More stable than additive methods

**Block-Diagonal Structure:**
Divides transformation into blocks:
- Each block is orthogonal matrix of size `oft_block_size`
- Independent transformations per block
- Reduces computation and parameters

### AWQ Compatibility

**Version Requirements:**
- **Minimum**: AutoAWQ 0.2.0
- Strict version checking prevents incompatibilities
- Clear error messages for version issues

**Quantized Base:**
- Base weights remain AWQ quantized
- Transformation applied to activations
- No impact on quantization scheme

**WQLinear_GEMM:**
Specific AWQ layer type for GEMM-optimized quantized operations.

### OFT Configuration Options

**Standard Parameters:**
- `oft_block_size`: Controls block granularity (default: 32)
- `r`: Rank parameter for transformations

**Advanced Options:**
- `coft`: Constrained OFT variant
- `use_cayley_neumann`: Cayley parameterization
- `num_cayley_neumann_terms`: Neumann series length
- `block_share`: Reuse blocks to reduce parameters

**Numerical Stability:**
- `eps`: Small constant (6e-5) for stability
- Important for orthogonalization process

### Dtype Management

**Conversion Pattern:**
- Checks if autocast enabled
- Converts input to oft_R weight dtype
- Transforms input
- Converts back to expected dtype
- Passes through quantized layer

**Careful Handling:**
Ensures compatibility between:
- OFT transformation dtype
- AWQ quantized operations
- Input/output expectations

## Usage Context
This adapter is dispatched when:
1. AutoAWQ (version >= 0.2.0) installed
2. Target is `WQLinear_GEMM` layer
3. OFT configuration applied

Useful for:
- Stable adaptation of AWQ models
- Property-preserving fine-tuning
- Continual learning scenarios

## Notes
- The `__repr__` method prefixes output with "oft."
- Version validation only at dispatch (not import) level
- Dual naming (`base_layer` and `quant_linear_module`) for compatibility
- Input transformation approach differs fundamentally from LoRA's additive approach
- Block structure essential for efficiency with large layers
- Cayley-Neumann provides theoretical orthogonality guarantees
- fan_in_fan_out handles different weight storage conventions
- Multiple dtype conversions ensure numerical correctness
