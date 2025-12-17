# Implementation: Intel Neural Compressor LoRA Linear Layer

## File Location
`src/peft/tuners/lora/inc.py`

## Overview
This module implements LoRA (Low-Rank Adaptation) for Intel Neural Compressor (INC) quantized linear layers. INC provides FP8 quantization optimized for Intel hardware, and this adapter enables fine-tuning of INC-quantized models.

## Key Components

### Class: `IncLoraLinear`
Applies LoRA adaptation to INC FP8 quantized linear layers.

**Conditional Definition:**
- Only defined if INC is available (`is_inc_available()`)

**Inheritance:**
- `Linear` (from peft.tuners.lora.layer)
- Inherits standard LoRA Linear implementation

**Key Features:**
- Minimal wrapper around standard Linear LoRA
- Explicitly disables merge/unmerge operations
- Designed for Intel Neural Compressor's FP8 quantization
- Testing handled in Optimum-Habana repository

**Constructor Parameters:**
- `base_layer` (torch.nn.Module): The INC quantized base layer
- `adapter_name` (str): Name of the adapter
- `**kwargs`: All standard LoRA parameters passed through

**Implementation:**
Simply delegates to parent `Linear` class constructor.

### Method: `merge(safe_merge, adapter_names)`
**Not Yet Implemented** - Raises `NotImplementedError`

**Error Message:**
"Merging LoRA with INC layers is not yet implemented"

**Rationale:**
- INC uses FP8 quantization format
- Merging logic not yet developed for this format
- Clear error message indicates future support planned

### Method: `unmerge()`
**Not Yet Implemented** - Raises `NotImplementedError`

**Error Message:**
"Unmerging LoRA from INC layers is not yet implemented"

**Rationale:**
- Companion to merge operation
- Cannot unmerge what hasn't been merged
- Placeholder for future implementation

### Function: `dispatch_inc(target, adapter_name, **kwargs)`
Factory function to create INC LoRA adapter when appropriate.

**Parameters:**
- `target` (torch.nn.Module): Target layer to potentially wrap
- `adapter_name` (str): Name for the adapter
- `**kwargs`: Additional configuration parameters

**Returns:**
- New `IncLoraLinear` instance if target is INC quantized, None otherwise

**Logic:**
1. Extracts base layer if target is already a tuner layer
2. Checks if INC is available
3. Imports `PatchedLinear` from Neural Compressor
4. Creates wrapper if target is `PatchedLinear`

**INC Layer Type:**
- Uses `neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules.PatchedLinear`
- This is INC's FP8-quantized linear layer implementation

## Dependencies
- `torch`
- `peft.import_utils.is_inc_available`
- `peft.tuners.tuners_utils.BaseTunerLayer`
- `peft.tuners.lora.layer.Linear`
- `neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules.PatchedLinear` (conditional)

## Key Characteristics

### Testing Strategy
**Important Note:** PEFT tests for INC are maintained in the Optimum-Habana repository:
- **LLMs**: https://github.com/huggingface/optimum-habana/blob/main/tests/test_peft_inference.py
- **Diffusers**: https://github.com/huggingface/optimum-habana/blob/main/tests/test_diffusers.py

This indicates tight integration with Intel's Habana Gaudi hardware ecosystem.

### Minimal Implementation
The implementation is intentionally minimal:
- Delegates most logic to standard `Linear` LoRA
- Only overrides methods that need INC-specific handling
- Currently only disables merge/unmerge

### Hardware Focus
- Designed for Intel Neural Compressor
- Optimized for Intel Habana Gaudi accelerators
- Uses FP8 quantization (efficient on specific hardware)

### Limitations
- **No Merge/Unmerge**: Operations not yet implemented
- **FP8 Specific**: Tied to INC's quantization format
- **Forward Only**: Can only apply adapters during forward pass

### Implementation Status
- Core functionality (forward pass): Complete
- Merge operations: Planned but not implemented
- Testing: Delegated to Optimum-Habana repository

## Usage Context
This adapter is automatically dispatched when:
1. Intel Neural Compressor is installed
2. Target layer is a `PatchedLinear` (INC's FP8 quantized layer)
3. LoRA configuration is applied

The integration with Optimum-Habana suggests this is primarily for users of Intel Gaudi hardware.

## Notes
- Copyright year 2025 indicates recent addition
- Testing delegation to Optimum-Habana is unusual but makes sense for hardware-specific code
- `PatchedLinear` name suggests INC patches standard PyTorch layers for quantization
- FP8 quantization is newer and more hardware-specific than int8 methods
- Future merge/unmerge support will likely require understanding INC's FP8 format
- The minimal implementation suggests INC quantization is compatible enough with standard LoRA that little adaptation is needed
