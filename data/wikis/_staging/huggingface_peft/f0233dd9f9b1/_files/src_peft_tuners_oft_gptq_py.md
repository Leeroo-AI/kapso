# File: `src/peft/tuners/oft/gptq.py`

**Category:** Quantization Backend

| Property | Value |
|----------|-------|
| Lines | 118 |
| Classes | `GPTQOFTLinear` |
| Functions | `dispatch_gptq` |
| Imports | peft, torch, typing |

## Understanding

**Status:** Fully explored

**Purpose:** Implements OFT adapter for GPTQ-quantized Linear layers, enabling efficient fine-tuning of GPTQ models.

**Mechanism:**

### GPTQOFTLinear
Applies OFT to GPTQ quantized layers:

**Forward**:
```python
if disable_adapters:
    return quant_linear_module(x)
for adapter in active_adapters:
    x = oft_R(x)  # Apply orthogonal transformation
result = quant_linear_module(x)
```

**Note**: Unlike bitsandbytes variants, merge/unmerge not supported (no merge methods)

### dispatch_gptq()
- Checks if target is GPTQModel QuantLinear or auto-gptq QuantLinear
- Creates GPTQOFTLinear wrapper
- Copies qweight reference

**Significance:** GPTQ is a popular quantization method for LLMs. This adapter enables fine-tuning GPTQ models with OFT without dequantizing base weights. The lack of merge support is by design - GPTQ's complex quantization makes merge unreliable. Users apply OFT transformations at runtime instead.

## Key Features

- **GPTQ Compatible**: Works with GPTQModel and auto-gptq
- **Runtime Transformation**: Applies OFT without modifying base weights
- **No Merge**: Avoids quantization artifacts

## Limitations

- **No Merge/Unmerge**: Not supported for GPTQ
- **Runtime Overhead**: Must apply transformation every forward pass
