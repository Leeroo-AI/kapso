# Implementation: HQQ OFT Linear Layer

## File Location
`src/peft/tuners/oft/hqq.py`

## Overview
This module implements OFT (Orthogonal Fine-Tuning) for HQQ (Half-Quadratic Quantization) quantized linear layers. Unlike most quantized OFT adapters, it provides full merge/unmerge support through HQQ's dequantization capabilities.

## Key Components

### Class: `HqqOFTLinear`
Full-featured OFT adapter for HQQ quantized linear layers.

**Conditional Definition:**
- Only defined if HQQ is available (`is_hqq_available()`)

**Inheritance:**
- `torch.nn.Module`
- `OFTLayer`

**Key Features:**
- Supports merge/unmerge operations (rare for quantized adapters)
- Orthogonal transformation of inputs
- Full quantization/dequantization workflow
- Block-diagonal structure
- Supports Cayley-Neumann parameterization
- COFT variant available

**Constructor Parameters:**
- `base_layer` (torch.nn.Module): HQQ quantized base layer
- `adapter_name` (str): Adapter name
- `r` (int): Rank parameter (default: 8)
- `oft_block_size` (int): Block size (default: 0)
- `module_dropout` (float): Dropout (default: 0.0)
- `init_weights` (bool): Initialize weights (default: True)
- `coft` (bool): Use constrained OFT (default: False)
- `eps` (float): Numerical stability (default: 6e-5)
- `block_share` (bool): Share blocks (default: False)
- `use_cayley_neumann` (bool): Cayley parameterization (default: False)
- `num_cayley_neumann_terms` (int): Neumann terms (default: 5)

**Attributes:**
- `fan_in_fan_out`: Set to False

### Method: `merge(safe_merge, adapter_names)`
Merges OFT adapters into HQQ quantized base weights.

**Unique Feature:**
Unlike most quantized OFT adapters, HQQ supports merging through:
1. Dequantizing base weights
2. Computing OFT rotation matrix
3. Applying matrix multiplication
4. Re-quantizing merged result

**Process:**
1. Validate adapters to merge
2. For each adapter:
   - Get base layer and save quantization config
   - Dequantize HQQ weights
   - Get OFT delta weight (rotation matrix)
   - Transpose weights: `output = transpose(output, 0, 1)`
   - Apply rotation: `w_data = torch.mm(oft_data, output)`
   - Transpose back: `w_data = transpose(w_data, 0, 1)`
   - Check for NaNs if safe_merge enabled
   - Create new HQQLinear layer
   - Quantize merged weights with original config
   - Replace base layer

**Matrix Multiplication:**
```python
output = torch.transpose(output, 0, 1)
w_data = torch.mm(oft_data, output.to(oft_data.dtype))
w_data = torch.transpose(w_data, 0, 1)
```

Note: The assignment `w_data = output.to(oft_data.dtype)` appears to overwrite the rotated result - possible bug.

### Method: `unmerge()`
Reverses merge operation, restoring original quantized weights.

**Process:**
1. Check if adapters merged
2. For each merged adapter (LIFO):
   - Dequantize current weights
   - Get OFT rotation matrix
   - Apply transpose of rotation: `oft_data.t()`
   - Re-quantize result

**Inverse Operation:**
Uses transpose of orthogonal matrix (inverse = transpose for orthogonal matrices).

### Method: `get_delta_weight(adapter)`
Retrieves the OFT rotation matrix.

**Returns:**
```python
return self.oft_R[adapter].get_weight()
```

Gets the learned orthogonal transformation from oft_R module.

### Method: `forward(x: torch.Tensor, *args, **kwargs)`
Forward pass with OFT adaptation.

**Implementation Flow:**
1. Check forward arguments
2. Pop `adapter_names` from kwargs (for mixed batch support)
3. Handle disable_adapters and merged states
4. For each active adapter:
   - Retrieve oft_R transformation
   - Handle dtype conversion
   - Apply transformation: `x = oft_R(x)`
5. Pass transformed input through base HQQ layer
6. Convert result to expected dtype if needed

**Mixed Batch Support:**
The `adapter_names` parameter suggests potential for per-sample adapters (not fully implemented here).

## Function: `dispatch_hqq(target, adapter_name, **kwargs)`
Factory function to create HQQ OFT adapter.

**Parameters:**
- `target` (torch.nn.Module): Target layer
- `adapter_name` (str): Adapter name
- `**kwargs`: Configuration parameters

**Returns:**
- New `HqqOFTLinear` instance if target is HQQ quantized, None otherwise

**Logic:**
- Extracts base layer if needed
- Checks for HQQLinear type
- Creates wrapper directly with base layer (not target)

## Dependencies
- `torch`
- `copy` - For deep copying quantization configs
- `warnings` - For merge/unmerge warnings
- `typing.Optional`
- `peft.import_utils.is_hqq_available`
- `peft.tuners.tuners_utils.{BaseTunerLayer, check_adapters_to_merge}`
- `peft.tuners.oft.layer.OFTLayer`
- `hqq.core.quantize.HQQLinear` (conditional)

## Key Characteristics

### Advanced Features

**Merge/Unmerge Support:**
Unique among quantized OFT adapters:
- Full dequantize-transform-requantize workflow
- Preserves quantization config
- Handles HQQ's offload_meta parameter

**HQQ Integration:**
- Preserves quantization config including `offload_meta`
- Maintains compute dtype and device placement
- Deep copies config to avoid mutation

### OFT with Quantization

**Orthogonal Transformation:**
- Applies rotation matrices to inputs
- Preserves norms and angles
- Block-diagonal structure

**Merge as Matrix Multiplication:**
For OFT, merging means:
```
W' = R @ W
```
where R is the learned orthogonal rotation.

**Transpose of Orthogonal Matrix:**
For unmerging, uses orthogonal property:
```
R^(-1) = R^T
W = R^T @ W'
```

### Potential Issues

**Merge Implementation:**
Line 97 appears problematic:
```python
output = torch.transpose(output, 0, 1)
w_data = torch.mm(oft_data, output.to(oft_data.dtype))
w_data = torch.transpose(w_data, 0, 1)
w_data = output.to(oft_data.dtype).to(oft_data.device)  # Overwrites w_data?
```

The last line may overwrite the rotated result.

### Memory Efficiency

**Base Weights:**
- Remain in HQQ quantized format
- Only OFT parameters in full precision

**Quantization Config:**
- Deep copied to avoid mutation
- `offload_meta` removed before quantization
- Restored after quantization complete

## Usage Context
This adapter is dispatched when:
1. HQQ library installed
2. Target is `HQQLinear`
3. OFT configuration applied

The comprehensive feature set makes HQQ one of the most flexible quantized OFT options.

## Notes
- The `__repr__` method prefixes output with "oft."
- Merge/unmerge operations leverage HQQ's dequantization
- Potential bug in merge method (line 97)
- Unmerge uses transpose instead of explicit inverse
- Mixed batch support partially implemented (adapter_names parameter)
- Quantization config requires special handling
- Block-diagonal structure for efficiency
- Cayley-Neumann parameterization for guaranteed orthogonality
- fan_in_fan_out not used (set to False)
