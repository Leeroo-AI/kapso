# File: `src/peft/tuners/lora/gptq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 154 |
| Classes | `GPTQLoraLinear` |
| Functions | `dispatch_gptq` |
| Imports | layer, peft, torch, typing |

## Understanding

**Status:** ✅ Documented

**Purpose:** LoRA adapter for GPTQ (Generative Pre-trained Transformer Quantization) quantized models

**Mechanism:** GPTQLoraLinear handles GPTQ's group-wise 4-bit quantization with learned rounding. Implements specialized forward pass that works with GPTQ's packed weight format and provides merge operations that dequantize using GPTQ's inverse quantization before applying LoRA updates.

**Significance:** Supports fine-tuning of GPTQ-quantized models, one of the most popular quantization methods for large language models. Critical for maintaining quantization benefits during adaptation, enabling 4-bit training of models like LLaMA while preserving memory efficiency and maintaining compatibility with GPTQ inference optimizations.
