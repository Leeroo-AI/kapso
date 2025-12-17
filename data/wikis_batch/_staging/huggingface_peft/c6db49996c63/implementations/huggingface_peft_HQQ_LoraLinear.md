# Implementation: HQQ LoRA Linear Layer

## File Location
`src/peft/tuners/lora/hqq.py`

## Overview
This module implements LoRA (Low-Rank Adaptation) for HQQ (Half-Quadratic Quantization) quantized linear layers. HQQ is a state-of-the-art quantization method, and this implementation provides comprehensive support including merge/unmerge operations and DoRA variants.

## Key Components

### Class: `HqqLoraLinear`
Full-featured LoRA adapter for HQQ quantized linear layers.

**Conditional Definition:**
- Only defined if HQQ is available (`is_hqq_available()`)

**Inheritance:**
- `torch.nn.Module`
- `LoraLayer`

**Key Features:**
- Supports merge/unmerge operations (unlike most quantized adapters)
- Supports DoRA (Weight-Decomposed Low-Rank Adaptation)
- Mixed batch inference with per-sample adapter selection
- Full quantization/dequantization workflow
- Does not support lora_bias (raises ValueError if enabled)

**Constructor Parameters:**
- `base_layer` (torch.nn.Module): The HQQ quantized base layer
- `adapter_name` (str): Name of the adapter
- `r` (int): Rank of the LoRA decomposition (default: 0)
- `lora_alpha` (int): Scaling factor for LoRA (default: 1)
- `lora_dropout` (float): Dropout probability (default: 0.0)
- `init_lora_weights` (bool): Whether to initialize LoRA weights (default: True)
- `use_rslora` (bool): Use rank-stabilized LoRA (default: False)
- `use_dora` (bool): Use DoRA variant (default: False)
- `lora_bias` (bool): Not supported, must be False

**Attributes:**
- `fan_in_fan_out`: Set to False (weight not transposed)

### Method: `resolve_lora_variant(use_dora, **kwargs)`
Determines whether to use DoRA variant.

**Returns:**
- `DoraLinearVariant()` if use_dora=True
- `None` for vanilla LoRA

### Method: `merge(safe_merge, adapter_names)`
Merges active adapter weights into HQQ quantized base weights.

**Unique Feature:**
Most quantized adapters don't support merging, but HQQ does through:
1. Dequantizing base weights
2. Computing and adding LoRA delta weights
3. Re-quantizing merged weights with original config

**Parameters:**
- `safe_merge` (bool): Check for NaNs before merging (default: False)
- `adapter_names` (list[str]): Specific adapters to merge (default: None = all active)

**Process:**
1. Validate adapters to merge
2. For each adapter:
   - Get base layer and save quantization config
   - Dequantize HQQ weights
   - Compute delta weight (vanilla LoRA or variant)
   - Check for NaNs if safe_merge enabled
   - Create new HQQLinear layer
   - Quantize merged weights with original config
   - Replace base layer

### Method: `unmerge()`
Reverses merge operation, restoring original quantized weights.

**Process:**
1. Check if any adapters are merged
2. For each merged adapter (LIFO order):
   - Dequantize current weights
   - Subtract LoRA delta weights
   - Re-quantize result
   - Replace base layer

### Method: `get_delta_weight(adapter)`
Computes the delta weight contribution from a LoRA adapter.

**Returns:**
- `torch.Tensor`: Delta weight computed as `(lora_B @ lora_A) * scaling`

**Implementation:**
- Uses transpose utility for proper weight orientation
- Applies adapter scaling factor

### Method: `_mixed_batch_forward(x, adapter_names, *args, **kwargs)`
Enables different adapters for different samples in the same batch.

**Advanced Feature:**
Allows heterogeneous adapter usage within a single forward pass.

**Parameters:**
- `x` (torch.Tensor): Input tensor
- `adapter_names` (list[str]): Adapter name for each sample in batch

**Process:**
1. Compute base layer output
2. Group samples by adapter name
3. For each unique adapter:
   - Extract sub-batch indices
   - Apply adapter only to relevant samples
   - Update corresponding output indices

**Use Case:**
Efficient inference when serving multiple adapter variants simultaneously.

### Method: `forward(x, *args, **kwargs)`
Main forward pass with comprehensive adapter handling.

**Execution Modes:**
1. **Disabled**: No adapter application
2. **Merged**: Use merged weights directly
3. **Mixed Batch**: Per-sample adapters (if `adapter_names` kwarg provided)
4. **Standard**: Apply active adapters sequentially

**Standard Mode Process:**
- Execute base HQQ quantized layer
- For each active adapter:
  - Handle dtype conversion
  - Apply vanilla LoRA or variant
  - Convert output back to expected dtype

## Function: `dispatch_hqq(target, adapter_name, **kwargs)`
Factory function to create HQQ LoRA adapter.

**Parameters:**
- `target` (torch.nn.Module): Target layer
- `adapter_name` (str): Adapter name
- `**kwargs`: Configuration parameters

**Returns:**
- New `HqqLoraLinear` instance if target is HQQ quantized, None otherwise

**Logic:**
- Extracts base layer if needed
- Checks for HQQLinear type
- Creates wrapper directly with base layer (not target)

## Dependencies
- `torch`
- `copy` - For deep copying quantization configs
- `warnings` - For merge/unmerge warnings
- `peft.import_utils.is_hqq_available`
- `peft.tuners.tuners_utils.{BaseTunerLayer, check_adapters_to_merge}`
- `peft.utils.other.transpose`
- `peft.tuners.lora.layer.{LoraLayer, LoraVariant}`
- `peft.tuners.lora.variants.DoraLinearVariant` (conditional)
- `hqq.core.quantize.HQQLinear` (conditional)

## Key Characteristics

### Advanced Features
1. **Full Merge/Unmerge Support**: Unique among quantized adapters
2. **DoRA Variant**: Weight-decomposed adaptation
3. **Mixed Batch Inference**: Different adapters per sample
4. **Safe Merging**: NaN detection before committing

### HQQ Integration
- Preserves quantization config including `offload_meta`
- Handles quantize/dequantize cycles transparently
- Maintains compute dtype and device placement

### Memory Efficiency
- Base weights in HQQ quantized format
- LoRA adapters in full precision
- Quantization config deep-copied to avoid mutation

### Dtype Management
- Checks autocast status for conversion decisions
- Casts input to adapter weight dtype when needed
- Preserves expected output dtype
- Handles CPU bf16/fp16 edge cases with fp32 casting

## Usage Context
This adapter is dispatched when:
1. HQQ library is installed
2. Target layer is `HQQLinear`
3. LoRA configuration applied

The comprehensive feature set makes HQQ one of the most flexible quantized adapter options in PEFT.

## Notes
- The `__repr__` method prefixes output with "lora."
- Mixed batch forward is particularly useful for serving scenarios
- Merge/unmerge operations may involve rounding errors
- Quantization config requires special handling (`offload_meta` removed before quantization)
- CPU bf16/fp16 operations cast to fp32 for matmul performance
