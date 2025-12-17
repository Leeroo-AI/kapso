# Implementation: Intel Neural Compressor OFT Linear Layer

## File Location
`src/peft/tuners/oft/inc.py`

## Overview
This module implements OFT (Orthogonal Fine-Tuning) for Intel Neural Compressor (INC) quantized linear layers. INC provides FP8 quantization optimized for Intel hardware, and this minimal adapter enables OFT on INC-quantized models.

## Key Components

### Class: `IncOFTLinear`
Applies OFT adaptation to INC FP8 quantized linear layers.

**Conditional Definition:**
- Only defined if INC is available (`is_inc_available()`)

**Inheritance:**
- `Linear` (from peft.tuners.oft.layer)
- Inherits standard OFT Linear implementation

**Key Features:**
- Minimal wrapper around standard Linear OFT
- Explicitly disables merge/unmerge operations
- Designed for Intel Neural Compressor's FP8 quantization
- Testing handled in Optimum-Habana repository

**Constructor Parameters:**
- `base_layer` (torch.nn.Module): INC quantized base layer
- `adapter_name` (str): Adapter name
- `**kwargs`: All standard OFT parameters passed through

**Implementation:**
Simply delegates to parent `Linear` class constructor.

### Method: `merge(safe_merge, adapter_names)`
**Not Yet Implemented** - Raises `NotImplementedError`

**Error Message:**
"Merging OFT with INC layers is not yet implemented"

**Rationale:**
- INC uses FP8 quantization format
- Merging logic not yet developed for this format
- Clear error indicates future support planned

### Method: `unmerge()`
**Not Yet Implemented** - Raises `NotImplementedError`

**Error Message:**
"Unmerging OFT from INC layers is not yet implemented"

**Rationale:**
- Companion to merge operation
- Placeholder for future implementation
- Cannot unmerge what hasn't been merged

### Function: `dispatch_inc(target, adapter_name, **kwargs)`
Factory function to create INC OFT adapter when appropriate.

**Parameters:**
- `target` (torch.nn.Module): Target layer
- `adapter_name` (str): Adapter name
- `**kwargs`: Configuration parameters

**Returns:**
- New `IncOFTLinear` instance if target is INC quantized, None otherwise

**Logic:**
1. Extracts base layer if target is already tuner layer
2. Checks if INC is available
3. Imports `PatchedLinear` from Neural Compressor
4. Creates wrapper if target is `PatchedLinear`

**INC Layer Type:**
Uses `neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules.PatchedLinear`

## Dependencies
- `torch`
- `typing.Optional`
- `peft.import_utils.is_inc_available`
- `peft.tuners.tuners_utils.BaseTunerLayer`
- `peft.tuners.oft.layer.Linear`
- `neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules.PatchedLinear` (conditional)

## Key Characteristics

### Testing Strategy

**External Testing:**
PEFT tests for INC are maintained in Optimum-Habana repository:
- **LLMs**: https://github.com/huggingface/optimum-habana/blob/main/tests/test_peft_inference.py
- **Diffusers**: https://github.com/huggingface/optimum-habana/blob/main/tests/test_diffusers.py

Indicates tight integration with Intel Habana Gaudi hardware.

### Minimal Implementation

**Design Philosophy:**
- Delegates most logic to standard `Linear` OFT
- Only overrides methods needing INC-specific handling
- Currently only disables merge/unmerge

**Why Minimal Works:**
- OFT transforms inputs (not weights)
- Compatible with INC quantization without special handling
- Forward pass uses standard OFT logic

### Hardware Focus

**Intel Optimization:**
- Designed for Intel Neural Compressor
- Optimized for Intel Habana Gaudi accelerators
- Uses FP8 quantization (efficient on specific hardware)

**FP8 Quantization:**
- Newer quantization format
- More hardware-specific than int8
- Tighter integration with Intel ecosystem

### Limitations

**Not Yet Implemented:**
1. **No Merge**: Operations not implemented
2. **No Unmerge**: Cannot reverse merge
3. **Forward Only**: Can only apply adapters during forward pass

**Future Work:**
Error messages indicate planned support for merge/unmerge.

### Implementation Status

**Current State:**
- Core functionality (forward pass): Complete
- Merge operations: Planned but not implemented
- Testing: Delegated to Optimum-Habana

**Forward Pass:**
Uses standard OFT logic from parent class:
- Orthogonal transformation of inputs
- Block-diagonal structure
- Cayley-Neumann parameterization support

## Usage Context

**Automatic Dispatch:**
This adapter is automatically selected when:
1. Intel Neural Compressor installed
2. Target is `PatchedLinear` (INC's FP8 quantized layer)
3. OFT configuration applied

**Target Users:**
- Intel Habana Gaudi hardware users
- Users with INC-quantized models
- Optimum-Habana ecosystem participants

## Notes

**File Header:**
- Copyright year 2025 indicates recent addition
- Same note structure as INC LoRA adapter

**Testing Delegation:**
- Unusual but sensible for hardware-specific code
- Optimum-Habana provides comprehensive testing
- Reduces duplication across repositories

**PatchedLinear:**
- Name suggests INC patches standard PyTorch layers
- Internal implementation detail of INC
- FP8 quantization applied through patching

**Similarity to INC LoRA:**
- Nearly identical structure to `src/peft/tuners/lora/inc.py`
- Same testing delegation
- Same merge/unmerge not implemented status
- Consistent approach across tuner types

**OFT Compatibility:**
- OFT's input transformation approach
- No weight access needed
- Natural fit for quantized layers
- Minimal adaptation required

**Future Development:**
- Merge/unmerge likely requires understanding INC's FP8 format
- May need special handling of PatchedLinear internals
- Optimum-Habana team likely to implement

## OFT with INC

**Orthogonal Transformation:**
Applied to inputs before INC quantized layer:
```
output = INC_layer(OFT(input))
```

**Benefits:**
- Preserves input properties
- Compatible with FP8 quantization
- Stable training dynamics
- Property-preserving adaptation

**Block-Diagonal Structure:**
- Standard OFT features available
- Configurable block size
- Parameter efficiency
- Computational efficiency
