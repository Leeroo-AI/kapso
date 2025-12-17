# Implementation: TorchAO LoRA Linear Layer

## File Location
`src/peft/tuners/lora/torchao.py`

## Overview
This module implements LoRA (Low-Rank Adaptation) for TorchAO (PyTorch Arithmetic Optimization) quantized linear layers. TorchAO provides low-level quantization primitives in PyTorch, and this adapter enables fine-tuning with sophisticated merge/unmerge support.

## Key Components

### Class: `TorchaoLoraLinear`
LoRA adapter for TorchAO quantized layers with full merge/unmerge capabilities.

**Inheritance:**
- `Linear` (from peft.tuners.lora.layer)
- Extends standard Linear LoRA implementation

**Key Features:**
- Supports merge/unmerge operations via dequantize/quantize cycles
- Currently limited to int8_weight_only quantization
- Uses TorchAO's tensor subclass system
- Handles memory constraints during merge operations
- Does not support lora_bias (raises ValueError if enabled)

**Constructor Parameters:**
- `*args`: Positional arguments for Linear
- `get_apply_tensor_subclass`: Function to get TorchAO tensor subclass config
- `**kwargs`: Standard LoRA parameters

**Additional Requirement:**
- `get_apply_tensor_subclass` must be provided as a kwarg
- This function is stored and used during merge/unmerge

**Initialization:**
1. Validates lora_bias is False
2. Calls parent Linear constructor
3. Stores get_apply_tensor_subclass function
4. Validates dtype is supported

### Method: `_check_dtype_supported()`
Validates that quantization dtype is supported.

**Current Limitation:**
Only int8 weights are supported.

**Validation Logic:**
Checks weight format for both TorchAO versions:
- **TorchAO 0.7.0+**: `weight.tensor_impl.data.dtype`
- **TorchAO < 0.7.0**: `weight.layout_tensor.data.dtype`

**Error:**
Raises `ValueError` if dtype is not torch.int8

### Method: `merge(safe_merge, adapter_names)`
Merges LoRA adapters into quantized base weights.

**Process:**
1. Validate adapters to merge
2. Check dtype support
3. Get base layer and current quantized weight
4. For each adapter:
   - Dequantize weights (may raise NotImplementedError)
   - Check for NaNs if safe_merge enabled
   - Add LoRA delta weight
   - Delete old weight (workaround for immutable tensors)
   - Assign new weight
   - Requantize using TorchAO's `quantize_` function
   - Clean up temporary weight
5. Track merged adapter

**TorchAO Workaround:**
Cannot directly mutate tensor data, must:
1. Delete weight attribute
2. Assign new weight
3. Call quantize_ to re-quantize

### Method: `unmerge()`
Reverses merge operation, restoring original quantized weights.

**Process:**
1. Check if any adapters merged
2. For each merged adapter (LIFO):
   - Dequantize current weights
   - Subtract LoRA delta weight
   - Delete old weight
   - Assign new weight
   - Requantize
   - Clean up

**Same Workaround:**
Uses delete-assign-quantize pattern due to TorchAO tensor immutability.

### Function: `dispatch_torchao(target, adapter_name, lora_config, **kwargs)`
Factory function to create TorchAO LoRA adapter.

**Parameters:**
- `target` (torch.nn.Module): Target layer
- `adapter_name` (str): Adapter name
- `lora_config` (LoraConfig): LoRA configuration object
- `**kwargs`: Additional parameters

**Returns:**
- New `TorchaoLoraLinear` instance if target uses TorchAO quantization, None otherwise

**Validation:**
1. Extracts base layer if needed
2. Checks if layer has weight attribute
3. Checks if TorchAO is available
4. Verifies weight is AffineQuantizedTensor or LinearActivationQuantizedTensor
5. Creates wrapper if all checks pass

**Supported TorchAO Types:**
- `AffineQuantizedTensor`: Standard quantized tensor
- `LinearActivationQuantizedTensor`: Layer with quantized activations

## Dependencies
- `torch`
- `warnings` - For merge/unmerge warnings
- `peft.import_utils.is_torchao_available`
- `peft.tuners.tuners_utils.{BaseTunerLayer, check_adapters_to_merge}`
- `peft.tuners.lora.config.LoraConfig`
- `peft.tuners.lora.layer.Linear`
- `torchao.quantize_` (function)
- `torchao.dtypes.AffineQuantizedTensor`
- `torchao.quantization.LinearActivationQuantizedTensor`

## Key Characteristics

### Merge/Unmerge Support
Unlike most quantized adapters, provides full merge/unmerge:
- Dequantizes weights
- Applies delta weights
- Re-quantizes with original config
- Handles TorchAO's immutable tensor constraints

### TorchAO Integration
- Uses TorchAO's `quantize_` function for re-quantization
- Respects tensor subclass system
- Works with multiple TorchAO versions (0.7.0+ and older)

### Current Limitations
1. **Int8 Only**: Currently only supports int8_weight_only quantization
2. **No lora_bias**: Bias not supported
3. **Dequantization Required**: Not all TorchAO quantization schemes support dequantization

### Memory Management
- Explicit cleanup with `del weight` after operations
- Temporary weight objects for merge operations
- Workaround for tensor immutability

### Error Handling
- Clear dtype validation errors
- Helpful messages about dequantization support
- Safe merge option for NaN detection

## Usage Context
This adapter is dispatched when:
1. TorchAO library is installed
2. Target layer uses TorchAO quantization (AffineQuantizedTensor or LinearActivationQuantizedTensor)
3. LoRA configuration applied

The get_apply_tensor_subclass function must be provided in kwargs for proper quantization during merge.

## Notes
- The `__repr__` method replaces "lora.Linear" with the actual class name
- Explicit deletion and reassignment of weights is a TorchAO-specific workaround
- Future support for other dtypes (int4, int2) may be added
- Version-specific checks handle TorchAO API changes
- TODO comment suggests future support for int4_weight_only
