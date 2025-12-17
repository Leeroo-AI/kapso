# File: `src/peft/tuners/oft/bnb.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 388 |
| Classes | `Linear8bitLt`, `Linear4bit` |
| Imports | __future__, bitsandbytes, layer, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** OFT adapter implementations for bitsandbytes 8-bit and 4-bit quantized layers

**Mechanism:** Linear8bitLt and Linear4bit classes wrap bnb quantized layers, dequantize for merge/unmerge operations, apply OFT rotation before quantized forward pass; dispatch functions detect and wrap appropriate bnb layers

**Significance:** Enables OFT finetuning on bitsandbytes quantized models with 8-bit and 4-bit precision, balancing memory efficiency with adaptation capability
