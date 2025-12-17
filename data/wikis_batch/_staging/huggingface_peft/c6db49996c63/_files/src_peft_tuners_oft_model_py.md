# File: `src/peft/tuners/oft/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 199 |
| Classes | `OFTModel` |
| Imports | aqlm, awq, eetq, gptq, hqq, inc, layer, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Top-level model wrapper that applies OFT adapters to pretrained models

**Mechanism:** Extends BaseTuner to create and replace target modules with OFT layers, dispatches to appropriate quantized variants (GPTQ, AQLM, AWQ, EETQ, BNB 8bit/4bit, HQQ, INC) or default layers based on base layer type; prevents merging for GPTQ and replicated layers

**Significance:** Main interface orchestrating OFT adapter injection with comprehensive quantization backend support for efficient deployment
