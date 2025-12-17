# Implementation: GPTQ OFT Linear Layer

## File Location
`src/peft/tuners/oft/gptq.py`

## Overview
This module implements OFT (Orthogonal Fine-Tuning) for GPTQ quantized linear layers. It supports both GPTQModel and AutoGPTQ libraries, applying orthogonal transformations to inputs of GPTQ-quantized layers.

## Key Components

### Class: `GPTQOFTLinear`
Applies OFT adaptation to GPTQ quantized linear layers.

**Inheritance:**
- `torch.nn.Module`
- `OFTLayer`

**Key Features:**
- Compatible with GPTQModel and AutoGPTQ
- Orthogonal transformation of layer inputs
- Block-diagonal structure for efficiency
- Supports Cayley-Neumann parameterization
- COFT variant available
- Block sharing capability

**Constructor Parameters:**
- `base_layer`: GPTQ quantized base layer
- `adapter_name` (str): Adapter name
- `r` (int): Rank parameter (default: 8)
- `oft_block_size` (int): Orthogonal block size (default: 0)
- `module_dropout` (float): Dropout probability (default: 0.0)
- `coft` (bool): Use constrained OFT (default: False)
- `eps` (float): Numerical stability (default: 6e-5)
- `block_share` (bool): Share blocks (default: False)
- `use_cayley_neumann` (bool): Cayley parameterization (default: False)
- `num_cayley_neumann_terms` (int): Neumann terms (default: 5)
- `fan_in_fan_out` (bool): Weight transpose (default: False)
- `init_weights` (bool): Initialize weights (default: True)

**Attributes:**
- `quant_linear_module`: Reference to GPTQ quantized layer

### Method: `forward(x: torch.Tensor)`
Forward pass with OFT adaptation on GPTQ quantized weights.

**Implementation Flow:**
1. **Execute quantized base layer first** (unusual pattern)
2. If adapters disabled, return quantized result
3. For each active adapter:
   - Retrieve oft_R orthogonal transformation
   - Handle dtype conversion when not in autocast
   - Apply transformation: `x = oft_R(x)`
4. **Execute quantized base layer again** with transformed input
5. Convert result to expected dtype if needed
6. Return result

**Unusual Pattern:**
```python
result = self.quant_linear_module(x)  # First execution (unused?)

if self.disable_adapters:
    return self.quant_linear_module(x)  # Second execution

# Transform input
for active_adapter in self.active_adapters:
    oft_R = self.oft_R[active_adapter]
    x = oft_R(x)

result = self.quant_linear_module(x)  # Third execution
return result
```

**Note:** The first `result` assignment appears redundant - result computed but not used.

### Function: `dispatch_gptq(target, adapter_name, **kwargs)`
Factory function supporting both GPTQModel and AutoGPTQ.

**Parameters:**
- `target` (torch.nn.Module): Target layer
- `adapter_name` (str): Adapter name
- `**kwargs`: Configuration including `gptq_quantization_config`

**Returns:**
- New `GPTQOFTLinear` instance if target is GPTQ quantized, None otherwise

**Dual Library Support:**

1. **GPTQModel** (checked first):
```python
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
if isinstance(target_base_layer, BaseQuantLinear):
    new_module = GPTQOFTLinear(target, adapter_name, **kwargs)
```

2. **AutoGPTQ** (fallback):
```python
quant_linear = get_auto_gptq_quant_linear(cfg)
if quant_linear is not None and isinstance(target_base_layer, quant_linear):
    new_module = GPTQOFTLinear(target, adapter_name, **kwargs)
```

**Setup:**
- Retrieves quantization config from kwargs
- Creates wrapper if library available
- Sets `qweight` attribute to base layer's quantized weights

## Dependencies
- `torch`
- `typing.{Any, Optional}`
- `peft.import_utils.is_gptqmodel_available`
- `peft.tuners.oft.layer.OFTLayer`
- `peft.tuners.tuners_utils.BaseTunerLayer`
- `peft.utils.get_auto_gptq_quant_linear`
- `gptqmodel.nn_modules.qlinear.BaseQuantLinear` (conditional)

## Key Characteristics

### GPTQ Integration

**Dual Library Support:**
- **GPTQModel**: Modern, actively maintained
- **AutoGPTQ**: Legacy support with multiple backends
- Graceful fallback ensures broad compatibility

**Quantized Base:**
- Base weights remain GPTQ quantized
- Transformation applied to activations
- No impact on quantization scheme

**Config-Based Dispatch:**
AutoGPTQ uses config to determine appropriate quantized linear class.

### OFT Application

**Orthogonal Input Transformation:**
Applies learned orthogonal matrices to inputs:
- Preserves norms and angles
- Stable training dynamics
- Property-preserving adaptation

**Block-Diagonal Structure:**
- Transformation divided into blocks
- Each block is orthogonal matrix of `oft_block_size`
- Reduces parameters and computation

### Configuration Options

**Standard Parameters:**
- `r`: Rank (default: 8, higher than other OFT implementations)
- `oft_block_size`: Block granularity (default: 0)

**Advanced Options:**
- `coft`: Constrained OFT variant
- `use_cayley_neumann`: Cayley parameterization for orthogonality
- `num_cayley_neumann_terms`: Neumann series approximation length
- `block_share`: Reuse blocks to reduce parameters

**Numerical Stability:**
- `eps`: Small constant (6e-5) for stability
- Critical for orthogonalization operations

### Potential Issue

**Redundant Computation:**
The forward method computes base layer result twice:
1. Line 63: `result = self.quant_linear_module(x)` (unused)
2. Line 81: `result = self.quant_linear_module(x)` (after transformation)

This may be a bug or dead code.

### Dtype Management

**Conversion Pattern:**
```python
requires_conversion = not torch.is_autocast_enabled()
if requires_conversion:
    expected_dtype = x.dtype
    x = self._cast_input_dtype(x, oft_R.weight.dtype)

x = oft_R(x)

result = self.quant_linear_module(x)
if requires_conversion:
    result = result.to(expected_dtype)
```

Ensures compatibility between OFT and GPTQ operations.

## Usage Context
This adapter is dispatched when:
1. GPTQModel or AutoGPTQ installed
2. Target is GPTQ quantized linear layer
3. OFT configuration applied

Useful for:
- Stable adaptation of GPTQ models
- Property-preserving fine-tuning
- Continual learning scenarios

## Notes
- The `__repr__` method prefixes output with "oft."
- Potential redundant computation in forward method (line 63)
- Dual library support ensures broad compatibility
- Config-based class lookup for AutoGPTQ flexibility
- Default rank (8) higher than AQLM/AWQ implementations (0)
- Dual naming (`base_layer` and `quant_linear_module`) for consistency
- fan_in_fan_out supports different weight storage conventions
- Block structure essential for large layer efficiency
- Cayley-Neumann provides theoretical orthogonality guarantee
