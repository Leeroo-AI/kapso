# File: `src/peft/tuners/lora/hqq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 251 |
| Classes | `HqqLoraLinear` |
| Functions | `dispatch_hqq` |
| Imports | __future__, copy, layer, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** HQQ quantized LoRA support

**Mechanism:** Implements LoRA for HQQ (Half-Quadratic Quantization) models. HqqLoraLinear wraps HQQLinear layers. Forward pass calls quantized base layer, adds LoRA contribution computed in full precision. Supports merge/unmerge by dequantizing weights, applying LoRA delta, creating new HQQLinear with updated weights, and requantizing using original quant_config. dispatch_hqq() detects HQQLinear and wraps appropriately. Supports DoRA variant. Does not support lora_bias yet.

**Significance:** Enables LoRA on HQQ-quantized models with full merge/unmerge support. HQQ provides flexible quantization with good quality-compression tradeoffs. Unlike most quantization backends, HQQ integration supports weight merging by requantization, making it more versatile for deployment scenarios where merged adapters are needed.
