# File: `src/peft/tuners/oft/eetq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 116 |
| Classes | `EetqOFTLinear` |
| Functions | `dispatch_eetq` |
| Imports | peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** OFT adapter implementation for EETQ-quantized linear layers

**Mechanism:** EetqOFTLinear wraps EETQ EetqLinear layers, applies OFT rotation before quantized computation; merge/unmerge operations raise NotImplementedError; dispatch_eetq detects EETQ layers

**Significance:** Enables OFT finetuning on EETQ-quantized models with inference-only support (no merging capability)
