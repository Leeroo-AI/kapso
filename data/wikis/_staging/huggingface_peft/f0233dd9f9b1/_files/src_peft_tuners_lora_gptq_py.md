# File: `src/peft/tuners/lora/gptq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 154 |
| Classes | `GPTQLoraLinear` |
| Functions | `dispatch_gptq` |
| Imports | layer, peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** GPTQ quantized LoRA support

**Mechanism:** Implements LoRA for GPTQ-quantized models. GPTQLoraLinear wraps GPTQ quantized linear layers (from auto-gptq or gptqmodel libraries). Forward pass calls quantized base layer, then adds LoRA contribution computed in higher precision. Does not support merging since GPTQ weights are stored in compressed format. dispatch_gptq() detects GPTQ layers (BaseQuantLinear or auto-gptq QuantLinear) and creates appropriate LoRA wrappers. Supports QALoRA variant for quantized adapters but explicitly disables DoRA due to incompatibility.

**Significance:** Enables LoRA fine-tuning on GPTQ-quantized models, which use advanced group-wise quantization for better quality than naive quantization. GPTQ is popular for deploying large models efficiently, so this integration makes fine-tuning practical for deployed quantized models. Important bridge between quantization and parameter-efficient fine-tuning.
