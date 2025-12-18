# File: `src/peft/tuners/oft/awq.py`

**Category:** Quantization Backend

| Property | Value |
|----------|-------|
| Lines | 119 |
| Classes | `AwqOFTLinear` |
| Functions | `dispatch_awq` |
| Imports | importlib, packaging, peft, torch, typing |

## Understanding

**Status:** Fully explored

**Purpose:** Implements OFT for AWQ (Activation-aware Weight Quantization) Linear layers.

**Mechanism:**

### AwqOFTLinear
Applies OFT to AWQ quantized WQLinear_GEMM layers:

**Forward**: Input transformation before AWQ layer
**No Merge**: Runtime transformation only

### dispatch_awq()
- Version check: Requires autoawq >= 0.2.0
- Checks for WQLinear_GEMM base layer
- Creates AwqOFTLinear wrapper

**Significance:** AWQ is activation-aware quantization. OFT adapters enable fine-tuning while preserving AWQ's calibration. Version checking ensures PEFT compatibility.

## Key Features

- **AWQ Support**: Works with autoawq library
- **Version Validation**: Ensures autoawq >= 0.2.0
- **Runtime Transform**: No merge, applies OFT during forward

## References

- AWQ Paper: Activation-aware Weight Quantization
- Minimum Version: autoawq 0.2.0
