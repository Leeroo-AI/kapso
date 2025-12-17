# File: `src/peft/tuners/oft/aqlm.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 105 |
| Classes | `AqlmOFTLinear` |
| Functions | `dispatch_aqlm` |
| Imports | peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** OFT adapter implementation for AQLM-quantized linear layers

**Mechanism:** AqlmOFTLinear wraps AQLM QuantizedLinear layers, applies OFT rotation before quantized forward pass; dispatch_aqlm function detects AQLM layers and creates appropriate wrapper

**Significance:** Enables OFT finetuning on AQLM-quantized models, maintaining memory efficiency while adding orthogonal adaptations
