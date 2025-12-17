# File: `src/peft/tuners/lora/aqlm.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 114 |
| Classes | `AqlmLoraLinear` |
| Functions | `dispatch_aqlm` |
| Imports | peft, torch, typing |

## Understanding

**Status:** ✅ Documented

**Purpose:** LoRA adapter for AQLM (Additive Quantization for Language Models) quantized weights

**Mechanism:** AqlmLoraLinear extends the base Linear layer to handle AQLM's multi-codebook quantization format. Implements dequantization before applying LoRA updates, then optionally requantizes. The dispatch_aqlm function detects AQLM quantized layers and wraps them appropriately.

**Significance:** Enables efficient fine-tuning of models compressed with AQLM quantization, which achieves high compression ratios through learned vector quantization. Critical for maintaining memory efficiency when adapting heavily quantized models.
