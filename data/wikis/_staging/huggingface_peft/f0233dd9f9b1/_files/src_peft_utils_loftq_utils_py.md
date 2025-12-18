# File: `src/peft/utils/loftq_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 410 |
| Classes | `NFQuantizer`, `_SafetensorLoader` |
| Functions | `loftq_init`, `replace_lora_weights_loftq` |
| Imports | __future__, accelerate, collections, huggingface_hub, logging, os, peft, safetensors, torch, transformers, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements LoftQ (Low-rank Quantization for Fine-tuning) to initialize LoRA weights for quantized models, minimizing quantization error through iterative decomposition.

**Mechanism:** NFQuantizer performs normal/uniform quantization to 2/4/8 bits with block-wise processing. loftq_init() iteratively quantizes weights, computes residuals, and SVD-decomposes them into low-rank LoRA matrices. replace_lora_weights_loftq() applies LoftQ on-the-fly by loading original weights from safetensors and computing optimal LoRA initialization. _SafetensorLoader handles sharded model loading.

**Significance:** Critical for high-quality quantized model fine-tuning. LoftQ significantly improves performance of LoRA on quantized models (4-bit) by initializing adapters to compensate for quantization error rather than using random initialization. Enables efficient fine-tuning of large quantized models with better starting point.
