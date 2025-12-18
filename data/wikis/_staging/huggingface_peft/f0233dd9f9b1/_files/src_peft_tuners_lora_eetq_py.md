# File: `src/peft/tuners/lora/eetq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 118 |
| Classes | `EetqLoraLinear` |
| Functions | `dispatch_eetq` |
| Imports | peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** EETQ quantized LoRA support

**Mechanism:** Implements LoRA for EETQ (Efficient Elastic Tensor Quantization) models. EetqLoraLinear wraps EETQ EetqLinear layers which use int8 quantization optimized for GPU inference. Forward pass calls quantized base layer, adds LoRA output with dtype conversion for autocast compatibility. Explicitly raises AttributeError for merge/unmerge operations as EETQ doesn't support weight modification. dispatch_eetq() detects EetqLinear layers and creates wrappers. Disables DoRA due to incompatibility with EETQ's quantization scheme.

**Significance:** Enables LoRA on EETQ-quantized models. EETQ provides fast int8 GPU inference with minimal accuracy loss. This integration allows fine-tuning of EETQ models while maintaining their inference speed advantages. Important for production scenarios where both efficiency and adaptability are required.
