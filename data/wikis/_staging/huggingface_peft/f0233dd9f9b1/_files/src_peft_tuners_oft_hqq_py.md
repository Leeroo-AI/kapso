# File: `src/peft/tuners/oft/hqq.py`

**Category:** Quantization Backend

| Property | Value |
|----------|-------|
| Lines | 186 |
| Classes | `HqqOFTLinear` |
| Functions | `dispatch_hqq` |
| Imports | __future__, copy, layer, peft, torch, typing, warnings |

## Understanding

**Status:** Fully explored

**Purpose:** Implements OFT for HQQ (Half-Quadratic Quantization) Linear layers with merge support.

**Mechanism:**

### HqqOFTLinear
Applies OFT to HQQ HQQLinear layers:

**Forward**: OFT transformation before HQQ layer

**Merge**:
- Dequantizes HQQ weights
- Applies OFT: `W_new = R @ W_old.T`
- Creates new HQQLinear with quantized merged weights
- Preserves quant_config and offload_meta

**Unmerge**: 
- Dequantizes, applies R^(-1), requantizes

### dispatch_hqq()
- Checks for hqq.HQQLinear base layer
- Creates HqqOFTLinear wrapper

**Significance:** HQQ uses half-quadratic optimization for quantization. Unlike other backends, HQQ OFT supports merge by creating new quantized layers. This is possible because HQQ exposes a clean quantization API.

## Key Features

- **HQQ Support**: Works with hqq library
- **Merge Support**: Can fold adapters (unlike most quantized backends)
- **Config Preservation**: Maintains quantization configuration
- **Safe Merge**: Optional NaN checking
