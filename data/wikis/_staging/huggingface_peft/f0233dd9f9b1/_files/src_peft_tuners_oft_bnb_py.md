# File: `src/peft/tuners/oft/bnb.py`

**Category:** Quantization Backend

| Property | Value |
|----------|-------|
| Lines | 388 |
| Classes | `Linear8bitLt`, `Linear4bit` |
| Imports | __future__, bitsandbytes, layer, peft, torch, typing, warnings |

## Understanding

**Status:** Fully explored

**Purpose:** Implements OFT adapters for bitsandbytes-quantized Linear layers (8-bit and 4-bit), enabling memory-efficient fine-tuning of quantized models.

**Mechanism:**

### Linear8bitLt (8-bit Quantization)
**Forward**: Applies OFT transformation to input before 8-bit layer
**Merge**: 
- Dequantizes base weights
- Applies orthogonal transformation: `W_new = R @ W_old.T`
- Requantizes to Int8Params
- Warning: May have rounding errors

**Unmerge**:
- Similar process with inverse transformation
- Warning: Numerical errors accumulate

### Linear4bit (4-bit Quantization)
**Forward**: Similar to 8-bit but with 4-bit quantized base
**Merge**:
- Dequantizes 4-bit weights
- Applies OFT transformation
- Requantizes to Params4bit with original config
- Handles bnb_quantized flag and compile attributes

**Unmerge**: Inverse transformation with 4-bit requantization

### Dispatch Functions
- `dispatch_bnb_8bit()`: Creates Linear8bitLt if target is bnb.nn.Linear8bitLt
- `dispatch_bnb_4bit()`: Creates Linear4bit if target is bnb.nn.Linear4bit

**Significance:** Enables OFT on quantized models, crucial for fine-tuning large models in memory-constrained environments. The implementation carefully handles quantization/dequantization cycles during merge operations. Warnings about rounding errors are honest about the precision-memory trade-off.

## Key Features

- **Memory Efficient**: OFT parameters in full precision, base weights quantized
- **Merge Support**: Can fold adapters into quantized weights
- **State Management**: Preserves bitsandbytes quantization state
- **Device Handling**: CPU-specific float32 casting for bf16/fp16

## Limitations

- **Merge Precision**: Rounding errors from quantization cycles
- **No Perfect Inverse**: Unmerge has numerical errors
