# File: `src/peft/tuners/adalora/gptq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 71 |
| Classes | `SVDQuantLinear` |
| Imports | layer, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** GPTQ quantized AdaLora layer

**Mechanism:** SVDQuantLinear extends AdaLoraLayer for GPTQ-quantized base layers. Implements forward pass with SVD adaptation (dropout(x) @ (lora_A * lora_E).T @ lora_B.T) * scaling / ranknum. Handles dtype conversion for autocast compatibility.

**Significance:** Enables AdaLora on GPTQ-quantized models. Essential for combining adaptive rank tuning with GPTQ quantization for memory-efficient fine-tuning.
