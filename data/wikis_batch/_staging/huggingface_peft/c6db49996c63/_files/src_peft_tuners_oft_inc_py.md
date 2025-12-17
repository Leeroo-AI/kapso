# File: `src/peft/tuners/oft/inc.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 78 |
| Classes | `IncOFTLinear` |
| Functions | `dispatch_inc` |
| Imports | layer, peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** OFT adapter implementation for Intel Neural Compressor (INC) quantized layers

**Mechanism:** IncOFTLinear extends standard OFT Linear for INC PatchedLinear layers, raises NotImplementedError for merge/unmerge; dispatch_inc detects INC FP8-quantized layers; tests located in Optimum-Habana repository

**Significance:** Enables OFT finetuning on Intel Neural Compressor FP8-quantized models for Habana accelerators, with limited functionality (no merging)
