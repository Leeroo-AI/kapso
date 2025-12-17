# Implementation: AQLM OFT Linear Layer

## File Location
`src/peft/tuners/oft/aqlm.py`

## Overview
This module implements OFT (Orthogonal Fine-Tuning) for AQLM (Accurate Quantized Low-rank Matrix) quantized linear layers. OFT learns orthogonal transformations that preserve model properties while adapting to new tasks, working seamlessly with AQLM's quantization scheme.

## Key Components

### Class: `AqlmOFTLinear`
Applies OFT adaptation to AQLM quantized linear layers.

**Inheritance:**
- `torch.nn.Module`
- `OFTLayer`

**Key Features:**
- Learns orthogonal transformation matrices
- Compatible with AQLM quantization
- Supports block-diagonal structure for efficiency
- Optional Cayley-Neumann parameterization
- COFT (Constrained OFT) variant support
- Block sharing across transformations

**Constructor Parameters:**
- `base_layer`: AQLM quantized base layer
- `adapter_name` (str): Adapter name
- `r` (int): Rank parameter (default: 0)
- `oft_block_size` (int): Size of orthogonal blocks (default: 32)
- `module_dropout` (float): Dropout probability (default: 0.0)
- `init_weights` (bool): Initialize OFT weights (default: True)
- `coft` (bool): Use constrained OFT (default: False)
- `eps` (float): Epsilon for numerical stability (default: 6e-5)
- `block_share` (bool): Share blocks across transformation (default: False)
- `fan_in_fan_out` (bool): Transpose weight representation (default: False)
- `use_cayley_neumann` (bool): Use Cayley-Neumann parameterization (default: False)
- `num_cayley_neumann_terms` (int): Terms in Cayley-Neumann expansion (default: 5)

**Initialization:**
Calls `update_layer` with all OFT parameters to initialize orthogonal transformation.

### Method: `forward(x: torch.Tensor)`
Performs forward pass with OFT adaptation on AQLM quantized weights.

**Key Characteristic:**
OFT transforms the **input** rather than adding to output (unlike LoRA).

**Implementation Flow:**
1. Return base layer output if adapters disabled
2. For each active adapter:
   - Retrieve oft_R (orthogonal transformation)
   - Handle dtype conversion when not in autocast
   - Apply transformation: `x = oft_R(x)`
3. Pass transformed input through base AQLM layer
4. Convert result to expected dtype if needed

**Transformation Order:**
```python
# Transform input first
for active_adapter in self.active_adapters:
    oft_R = self.oft_R[active_adapter]
    x = oft_R(x)

# Then apply base layer
result = self.base_layer(x)
```

**Difference from LoRA:**
- LoRA: `output = base(x) + lora(x)`
- OFT: `output = base(oft(x))`

### Function: `dispatch_aqlm(target, adapter_name, **kwargs)`
Factory function to create AQLM OFT adapter when appropriate.

**Parameters:**
- `target` (torch.nn.Module): Target layer
- `adapter_name` (str): Adapter name
- `**kwargs`: OFT configuration parameters

**Returns:**
- New `AqlmOFTLinear` instance if target is AQLM quantized, None otherwise

**Logic:**
1. Extracts base layer if target is already tuner layer
2. Checks if AQLM is available
3. Verifies target is AQLM `QuantizedLinear`
4. Creates wrapper if conditions met
5. Sets `qweight` attribute to base layer's codes

## Dependencies
- `torch`
- `typing.{Any, Optional}`
- `peft.import_utils.is_aqlm_available`
- `peft.tuners.oft.layer.OFTLayer`
- `peft.tuners.tuners_utils.BaseTunerLayer`
- `aqlm.QuantizedLinear` (conditional import)

## Key Characteristics

### OFT Method

**Orthogonal Transformations:**
OFT learns orthogonal matrices that:
- Preserve vector norms
- Maintain angular relationships
- Ensure stable training dynamics
- Prevent catastrophic forgetting

**Block-Diagonal Structure:**
- Transformation divided into blocks of size `oft_block_size`
- Each block is an independent orthogonal matrix
- Reduces parameters and computation
- Balances expressiveness and efficiency

**Mathematical Formulation:**
```
R = block_diag(R_1, R_2, ..., R_k)
where each R_i is orthogonal: R_i^T R_i = I
```

### OFT Variants

**Standard OFT:**
Learns full orthogonal transformations within blocks.

**COFT (Constrained OFT):**
- Additional constraints on transformations
- Controlled by `coft` parameter
- More stable for certain tasks

**Cayley-Neumann Parameterization:**
- Uses Cayley transform: `R = (I + A)(I - A)^(-1)`
- Approximated with Neumann series
- Number of terms controlled by `num_cayley_neumann_terms`
- Ensures orthogonality by construction

**Block Sharing:**
- Reuse same orthogonal block multiple times
- Reduces parameters further
- Controlled by `block_share` parameter

### AQLM Integration

**Quantized Base:**
- Base weights remain in AQLM quantized format
- OFT transformation applied to activations, not weights
- Preserves memory benefits of quantization

**Dtype Handling:**
- Checks autocast status
- Casts input to oft_R weight dtype
- Ensures result matches expected dtype
- Compatible with AQLM's int-based representation

### Limitations
- **No Merge Support**: Merging not mentioned (likely not supported)
- **Input Transformation**: Must transform entire input (can't be selective like additive methods)
- **Orthogonality Constraint**: Less flexible than unconstrained methods

## Usage Context
This adapter is automatically dispatched when:
1. AQLM library is installed
2. Target layer is AQLM `QuantizedLinear`
3. OFT configuration is applied

Particularly useful for:
- Tasks requiring property preservation
- Continual learning scenarios
- Stable adaptation to new domains

## Notes
- The `__repr__` method prefixes output with "oft."
- Logic differs from LoRA: transforms input instead of adding to output
- `qweight` set to `codes` attribute (AQLM-specific)
- Orthogonal constraints provide theoretical guarantees
- Block structure critical for computational efficiency
- Cayley-Neumann provides elegant orthogonality guarantee
- COFT and block sharing provide additional efficiency/stability options
- fan_in_fan_out parameter allows compatibility with transposed weight layouts
- Epsilon parameter (6e-5) provides numerical stability in orthogonalization
