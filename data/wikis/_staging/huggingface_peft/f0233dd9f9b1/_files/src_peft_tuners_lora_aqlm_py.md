# File: `src/peft/tuners/lora/aqlm.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 114 |
| Classes | `AqlmLoraLinear` |
| Functions | `dispatch_aqlm` |
| Imports | peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** AQLM quantized LoRA support

**Mechanism:** Implements LoRA for AQLM (Additive Quantization of Language Models) quantized models. AqlmLoraLinear wraps AQLM QuantizedLinear layers. Forward pass executes quantized base layer, then adds LoRA contribution (B @ A @ dropout(x)) * scaling computed in higher precision with automatic dtype conversion. Does not support merge/unmerge due to AQLM's complex additive quantization scheme. dispatch_aqlm() detects QuantizedLinear layers and wraps them. Explicitly disables DoRA as it's incompatible with AQLM quantization format.

**Significance:** Enables LoRA on AQLM-quantized models. AQLM uses sophisticated multi-codebook additive quantization for extreme compression with minimal quality loss. This integration makes fine-tuning practical for heavily compressed models, extending PEFT capabilities to cutting-edge quantization methods.
