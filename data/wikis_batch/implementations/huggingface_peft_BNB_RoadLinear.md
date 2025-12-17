# Implementation: BitsAndBytes Quantized RoAd Linear Layers

## File Location
`src/peft/tuners/road/bnb.py`

## Overview
This module implements RoAd (Robust Adaptation) for BitsAndBytes quantized linear layers. RoAd uses orthogonal transformations applied directly to layer outputs, with trainable parameters theta and alpha controlling the rotation and scaling.

## Key Components

### Class: `Linear8bitLt`
RoAd implementation for 8-bit quantized linear layers.

**Conditional Definition:**
- Only defined if BitsAndBytes is available (`is_bnb_available()`)

**Inheritance:**
- `torch.nn.Module`
- `RoadLayer`

**Key Features:**
- Applies orthogonal rotations to layer outputs
- Supports multiple RoAd variants (road_1, road_2, etc.)
- Group-based transformations
- Supports merge/unmerge operations
- Works with 8-bit quantization and bias

**Constructor Parameters:**
- `base_layer` (torch.nn.Module): BitsAndBytes 8-bit quantized layer
- `adapter_name` (str): Adapter name
- `variant` (RoadVariant): RoAd variant type (default: "road_1")
- `group_size` (int): Size of orthogonal transformation groups (default: 64)
- `init_weights` (bool): Initialize trainable weights (default: True)

**Key Attributes:**
- `variant`: Determines transformation strategy
- `group_size`: Controls granularity of rotations

### Method: `merge(safe_merge, adapter_names)` (Linear8bitLt)
Merges RoAd adapters into 8-bit quantized base weights.

**Warning:**
"Merge road module to 8-bit linear may get different generations due to rounding errors."

**Process:**
1. Validate adapters to merge
2. For each adapter:
   - Dequantize base weights
   - Compute rotation matrix R: `_get_delta_weight(variant, group_size, theta, alpha)`
   - Apply rotation: `R @ W` (matrix multiplication)
   - Handle bias if present: `R @ bias`
   - Check for NaNs if safe_merge enabled
   - Re-quantize as Int8Params
   - Reset gradients
   - Track merged adapter

**Bias Handling:**
If layer has bias:
```python
bias_data = bias.data
new_bias = torch.matmul(road_R, bias_data.to(road_R.dtype))
bias.data = new_bias.to(orig_dtype)
```

### Method: `unmerge()` (Linear8bitLt)
Reverses merge operation for 8-bit layers.

**Process:**
1. Check if adapters merged
2. For each merged adapter (LIFO):
   - Dequantize current weights
   - Compute rotation matrix R
   - Compute inverse: `inv_R = torch.linalg.inv(R.to(torch.float32))`
   - Apply inverse rotation: `inv_R @ W`
   - Handle bias: `inv_R @ bias`
   - Re-quantize
   - Reset gradients

**Inverse Computation:**
Uses `torch.linalg.inv` to compute matrix inverse, cast to float32 for stability.

### Method: `forward(x: torch.Tensor, *args, **kwargs)` (Linear8bitLt)
Forward pass with RoAd adaptation.

**Implementation Flow:**
1. Handle disable_adapters and merged states
2. Execute base layer
3. For each active adapter:
   - Handle dtype conversion
   - Apply RoAd transformation: `_apply_road(variant, group_size, theta, alpha, result)`
   - Convert back to expected dtype

**Transformation Application:**
RoAd applies rotation directly to layer output:
```python
result = _apply_road(
    self.variant[active_adapter],
    self.group_size[active_adapter],
    self.road_theta[active_adapter],
    self.road_alpha[active_adapter],
    result
)
```

### Function: `dispatch_bnb_8bit(target, adapter_name, **kwargs)`
Factory function for creating 8-bit RoAd adapters.

**Parameters:**
- `target` (torch.nn.Module): Target layer
- `adapter_name` (str): Adapter name
- `**kwargs`: Configuration including loaded_in_8bit flag

**Returns:**
- New `Linear8bitLt` instance if target is 8-bit quantized, None otherwise

**8-bit Specific Configuration:**
Copies kwargs and adds:
- `has_fp16_weights`: From target state
- `threshold`: From target state
- `index`: From target

### Class: `Linear4bit`
RoAd implementation for 4-bit quantized linear layers.

**Conditional Definition:**
- Only defined if BitsAndBytes 4-bit is available (`is_bnb_4bit_available()`)

**Key Features:**
- Same RoAd approach as 8-bit
- Works with 4-bit quantization
- More memory efficient
- Handles bias transformations

**Constructor:**
Identical to `Linear8bitLt` with same parameters.

### Method: `merge(safe_merge, adapter_names)` (Linear4bit)
Merges RoAd adapters into 4-bit quantized weights.

**Process:**
Similar to 8-bit but uses 4-bit specific operations:
1. Dequantize with `dequantize_bnb_weight(weight, state=weight.quant_state)`
2. Apply rotation matrix
3. Handle kwargs cleanup (remove '_' prefixed attributes from torch.compile)
4. Re-quantize as Params4bit

**Kwargs Cleanup:**
```python
if "bnb_quantized" in kwargs:
    kwargs["bnb_quantized"] = False
kwargs["requires_grad"] = False
kwargs.pop("data", None)
kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
```

### Method: `forward(x: torch.Tensor, *args, **kwargs)` (Linear4bit)
Forward pass for 4-bit with RoAd.

**Note in Code:**
Commented out defensive clone:
```python
# result = result.clone()
```
Suggests it may not be needed for RoAd (unlike LoRA).

**Otherwise identical to 8-bit version.**

### Function: `dispatch_bnb_4bit(target, adapter_name, **kwargs)`
Factory function for 4-bit RoAd adapters.

**4-bit Specific Configuration:**
Copies kwargs and adds:
- `compute_dtype`: From target base layer
- `compress_statistics`: From weight
- `quant_type`: From weight

## Helper Functions (from .layer)

### `_get_delta_weight(variant, group_size, theta, alpha)`
Computes rotation matrix R from trainable parameters.

**Parameters:**
- `variant`: RoAd variant type
- `group_size`: Size of rotation groups
- `theta`: Rotation angles
- `alpha`: Scaling factors

**Returns:**
Block-diagonal rotation matrix.

### `_apply_road(variant, group_size, theta, alpha, x)`
Applies RoAd transformation to tensor.

**Efficient forward pass implementation:**
- Avoids materializing full rotation matrix
- Applies transformations in grouped manner

## Dependencies
- `torch`
- `warnings`
- `bitsandbytes as bnb`
- `peft.import_utils.{is_bnb_available, is_bnb_4bit_available}`
- `peft.tuners.tuners_utils.{BaseTunerLayer, check_adapters_to_merge}`
- `peft.utils.integrations.dequantize_bnb_weight`
- `peft.tuners.road.config.RoadVariant`
- `peft.tuners.road.layer.{RoadLayer, _apply_road, _get_delta_weight}`

## Key Characteristics

### RoAd Method

**Orthogonal Transformations:**
- Applies rotation matrices to layer outputs
- Preserves output magnitude (orthogonal)
- More stable than additive methods
- Group-based for efficiency

**Trainable Parameters:**
- `road_theta`: Rotation angles for each group
- `road_alpha`: Scaling factors for each group
- Much fewer parameters than LoRA

**Variants:**
Different RoAd variants (road_1, road_2, etc.) provide:
- Different rotation strategies
- Varying expressiveness/efficiency tradeoffs

### Group-Based Design
**Group Size:**
- Controls granularity of rotations
- Smaller groups: More expressive, more parameters
- Larger groups: Fewer parameters, less expressive
- Default: 64

**Block-Diagonal Structure:**
Rotation matrix is block-diagonal:
- Each group has independent rotation
- Efficient computation and memory

### Merge/Unmerge Support
Supports merging via matrix multiplication:
- Forward: `W' = R @ W`
- Bias: `b' = R @ b`
- Inverse: `W = R^(-1) @ W'`

**Orthogonal Property:**
Inverse computation is stable because rotations are orthogonal.

### Quantization Integration

**8-Bit:**
- Uses `dequantize_bnb_weight` helper
- Int8Params for re-quantization
- State management with SCB

**4-Bit:**
- Uses `dequantize_4bit` function
- Params4bit for re-quantization
- Kwargs cleanup for torch.compile compatibility

## Usage Context
These classes are used when:
1. BitsAndBytes quantization applied
2. RoAd configuration specified
3. Need for stable, rotation-based adaptation

Particularly effective for:
- Tasks requiring output distribution preservation
- Stable training dynamics
- Low parameter count requirements

## Notes
- Both classes use `__repr__` prefix "road."
- Merge warnings about rounding errors
- Inverse rotation computed at float32 for numerical stability
- Commented defensive clone in 4-bit suggests different backprop behavior than LoRA
- Bias transformation is seamless (same rotation as weights)
- Group-based design balances expressiveness and efficiency
- Orthogonal constraints provide training stability
- torch.compile compatibility handled with kwargs cleanup
