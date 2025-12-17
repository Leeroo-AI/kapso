# File: `src/peft/tuners/oft/awq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 119 |
| Classes | `AwqOFTLinear` |
| Functions | `dispatch_awq` |
| Imports | importlib, packaging, peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** OFT adapter implementation for AWQ-quantized linear layers

**Mechanism:** AwqOFTLinear wraps AWQ WQLinear_GEMM layers, applies OFT rotation before quantized computation; dispatch_awq detects AWQ layers and checks version compatibility (requires autoawq >= 0.2.0)

**Significance:** Enables OFT finetuning on AWQ-quantized models, combining activation-aware quantization with orthogonal parameter-efficient adaptation
