# File: `src/peft/tuners/randlora/bnb.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 456 |
| Classes | `Linear8bitLt`, `Linear4bit` |
| Imports | __future__, bitsandbytes, layer, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Provides RandLoRA implementations for bitsandbytes quantized layers (8-bit and 4-bit), enabling RandLoRA with quantized base models.

**Mechanism:** Both classes extend RandLoraLayer and implement merge/unmerge by dequantizing weights, applying RandLoRA transformations (B @ A scaling), and re-quantizing. The get_scaled_bases method handles dtype casting for CPU bf16/fp16. Forward pass applies the scaled random bases directly to inputs without dequantization. Merge/unmerge operations warn about potential rounding errors from quantization.

**Significance:** Enables using RandLoRA with quantized models (QLoRA-style), combining the parameter efficiency of RandLoRA with memory savings from quantization. Critical for running large models on limited hardware. The implementation carefully handles quantization state and dtype conversions for compatibility.
