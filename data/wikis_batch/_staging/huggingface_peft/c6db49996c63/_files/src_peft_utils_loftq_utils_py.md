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

**Purpose:** Implements LoftQ (Low-rank Quantization-aware Fine-tuning) initialization for LoRA adapters on quantized models.

**Mechanism:** Uses iterative quantization and SVD decomposition to jointly optimize quantized base weights and LoRA adapter initialization, minimizing reconstruction error. Includes NFQuantizer for NormalFloat quantization and utilities to replace LoRA weights with LoftQ-initialized values.

**Significance:** Specialized initialization technique that significantly improves LoRA performance on quantized models by accounting for quantization errors during adapter initialization, bridging the gap between full-precision and quantized fine-tuning.
