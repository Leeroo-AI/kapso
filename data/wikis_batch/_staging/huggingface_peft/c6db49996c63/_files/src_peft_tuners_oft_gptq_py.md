# File: `src/peft/tuners/oft/gptq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 118 |
| Classes | `GPTQOFTLinear` |
| Functions | `dispatch_gptq` |
| Imports | peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** OFT adapter implementation for GPTQ-quantized linear layers

**Mechanism:** GPTQOFTLinear wraps GPTQ QuantLinear layers (from gptqmodel or auto-gptq), applies OFT rotation before quantized forward pass; dispatch_gptq detects GPTQ layers from either backend

**Significance:** Enables OFT finetuning on GPTQ-quantized models, supporting both GPTQModel and AutoGPTQ quantization backends
