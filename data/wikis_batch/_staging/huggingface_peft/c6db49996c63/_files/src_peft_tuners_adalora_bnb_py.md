# File: `src/peft/tuners/adalora/bnb.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 143 |
| Classes | `SVDLinear8bitLt`, `SVDLinear4bit` |
| Imports | layer, peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements AdaLoRA layers for quantized models using bitsandbytes (8-bit and 4-bit)

**Mechanism:** SVDLinear8bitLt and SVDLinear4bit classes wrap quantized base layers, applying SVD-based adaptation via lora_A, lora_B, and lora_E parameters. Forward passes compute output = base(x) + dropout(x) @ (lora_A * lora_E).T @ lora_B.T * scaling/ranknum with dtype conversions for quantized weights

**Significance:** Enables AdaLoRA's adaptive low-rank fine-tuning on memory-constrained systems by supporting bitsandbytes quantization, allowing efficient training of large models with dynamic rank allocation
