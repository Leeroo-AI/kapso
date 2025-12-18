# File: `src/peft/tuners/oft/aqlm.py`

**Category:** Quantization Backend

| Property | Value |
|----------|-------|
| Lines | 105 |
| Classes | `AqlmOFTLinear` |
| Functions | `dispatch_aqlm` |
| Imports | peft, torch, typing |

## Understanding

**Status:** Fully explored

**Purpose:** Implements OFT for AQLM (Additive Quantization of Language Models) quantized layers.

**Mechanism:**

### AqlmOFTLinear
Applies OFT to AQLM QuantizedLinear layers:

**Forward**: OFT transformation before AQLM quantized layer
**No Merge**: Runtime only

### dispatch_aqlm()
- Checks for aqlm.QuantizedLinear base layer
- Creates AqlmOFTLinear wrapper
- Copies qweight (codes) reference

**Significance:** AQLM uses additive quantization with codebooks for extreme compression. OFT adapters enable fine-tuning these heavily compressed models.

## Key Features

- **AQLM Support**: Works with aqlm library
- **Codebook Preservation**: Maintains quantized structure
- **Memory Efficient**: OFT params in FP, base quantized
