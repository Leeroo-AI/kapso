# File: `src/peft/tuners/adalora/gptq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 71 |
| Classes | `SVDQuantLinear` |
| Imports | layer, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** AdaLoRA layer implementation for GPTQ-quantized models

**Mechanism:** SVDQuantLinear wraps GPTQ quantized base layers and applies SVD-based adaptation. Forward pass: result = quant_linear(x) + (dropout(x) @ (lora_A * lora_E).T @ lora_B.T) * scaling/ranknum with dtype casting for float32 computation when not in autocast mode

**Significance:** Extends AdaLoRA support to GPTQ quantization (alternative to bitsandbytes), enabling adaptive rank fine-tuning on models compressed with different quantization schemes for broader hardware compatibility
