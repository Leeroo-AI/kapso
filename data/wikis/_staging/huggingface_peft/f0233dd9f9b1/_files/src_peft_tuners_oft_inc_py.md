# File: `src/peft/tuners/oft/inc.py`

**Category:** Quantization Backend

| Property | Value |
|----------|-------|
| Lines | 78 |
| Classes | `IncOFTLinear` |
| Functions | `dispatch_inc` |
| Imports | layer, peft, torch, typing |

## Understanding

**Status:** Fully explored

**Purpose:** Implements OFT for Intel Neural Compressor (INC) FP8-quantized layers.

**Mechanism:**

### IncOFTLinear
Inherits from standard OFT Linear:

**Merge/Unmerge**: Raises NotImplementedError (work in progress)

### dispatch_inc()
- Checks for neural_compressor.torch PatchedLinear
- Creates IncOFTLinear wrapper

**Testing**: Tests handled in Optimum-Habana repository

**Significance:** Intel NC provides FP8 quantization for Intel hardware. OFT support is early-stage with merge operations not yet implemented. Testing is external to PEFT.

## Key Features

- **INC Support**: Works with Intel Neural Compressor
- **FP8 Quantization**: Intel hardware optimization
- **Forward Only**: Merge/unmerge not yet implemented

## Testing

- LLM tests: optimum-habana/tests/test_peft_inference.py
- Diffuser tests: optimum-habana/tests/test_diffusers.py

## Status

- **Merge**: Not implemented (raises NotImplementedError)
- **Forward**: Fully functional
