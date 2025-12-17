# File: `src/peft/tuners/oft/hqq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 186 |
| Classes | `HqqOFTLinear` |
| Functions | `dispatch_hqq` |
| Imports | __future__, copy, layer, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** OFT adapter implementation for HQQ-quantized linear layers

**Mechanism:** HqqOFTLinear wraps HQQ HQQLinear layers, dequantizes weights for merge/unmerge operations creating new HQQLinear instances with modified weights, applies OFT rotation before quantized forward pass; dispatch_hqq detects HQQ layers

**Significance:** Enables OFT finetuning on HQQ-quantized models with full merge/unmerge support through weight requantization
