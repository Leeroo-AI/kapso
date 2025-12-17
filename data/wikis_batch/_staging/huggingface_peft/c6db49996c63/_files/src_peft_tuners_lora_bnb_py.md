# File: `src/peft/tuners/lora/bnb.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 611 |
| Classes | `Linear8bitLt`, `Linear4bit` |
| Imports | __future__, bitsandbytes, config, layer, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** LoRA adapters for BitsAndBytes (bnb) 8-bit and 4-bit quantized models

**Mechanism:** Implements Linear8bitLt and Linear4bit classes that wrap bitsandbytes quantized layers. Handles special forward logic for quantized computation, manages adapter dtype compatibility with quantized weights, and provides merge/unmerge operations that work with bnb's quantization states. Includes dispatch functions for both 8-bit and 4-bit variants.

**Significance:** Core component enabling QLoRA (Quantized LoRA) - the widely-used technique for fine-tuning large language models in drastically reduced memory. Makes it possible to fine-tune 65B+ parameter models on consumer GPUs by keeping base weights in 4-bit precision.
