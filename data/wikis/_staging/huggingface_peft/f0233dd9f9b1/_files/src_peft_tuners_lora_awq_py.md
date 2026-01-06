# File: `src/peft/tuners/lora/awq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 121 |
| Classes | `AwqLoraLinear` |
| Functions | `dispatch_awq` |
| Imports | importlib, packaging, peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** AWQ quantized LoRA support

**Mechanism:** Implements LoRA for AWQ (Activation-aware Weight Quantization) models. AwqLoraLinear wraps AWQ quantized linear layers (WQLinear_GEMM from autoawq). Forward pass calls quantized base layer, then adds LoRA contribution computed in full precision with dtype conversion handling. Does not support merge/unmerge operations due to AWQ's specialized quantization format. dispatch_awq() detects AWQ layers and creates LoRA wrappers, with version checking to ensure autoawq >= 0.2.0 for compatibility. Explicitly disables DoRA variant as it's incompatible with AWQ quantization.

**Significance:** Enables LoRA fine-tuning on AWQ-quantized models. AWQ is an advanced quantization method that preserves activation patterns for better quality, popular for efficient inference. This integration allows users to fine-tune AWQ-compressed models without dequantization, maintaining memory efficiency while enabling adaptation.
