# File: `src/peft/tuners/lora/awq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 121 |
| Classes | `AwqLoraLinear` |
| Functions | `dispatch_awq` |
| Imports | importlib, packaging, peft, torch, typing |

## Understanding

**Status:** ✅ Documented

**Purpose:** LoRA adapter for AWQ (Activation-aware Weight Quantization) quantized models

**Mechanism:** AwqLoraLinear handles AWQ's 4-bit quantization format with per-channel scaling. Overrides merge/unmerge operations to properly dequantize AWQ weights, apply LoRA deltas, and requantize. Checks for AutoAWQ library availability and version compatibility.

**Significance:** Supports fine-tuning of models quantized with AWQ, which optimizes quantization based on activation patterns to preserve accuracy. Essential for memory-efficient adaptation of AWQ-compressed language models while maintaining the benefits of quantization.
