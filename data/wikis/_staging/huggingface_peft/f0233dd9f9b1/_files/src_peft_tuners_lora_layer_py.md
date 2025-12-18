# File: `src/peft/tuners/lora/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 2304 |
| Classes | `LoraVariant`, `LoraLayer`, `Linear`, `Embedding`, `_ConvNd`, `Conv2d`, `Conv1d`, `Conv3d`, `MultiheadAttention`, `_LoraParameterProxy`, `ParamWrapper` |
| Functions | `dispatch_default` |
| Imports | __future__, config, contextlib, math, peft, torch, transformers, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core LoRA layer implementations

**Mechanism:** Implements the fundamental LoRA layer classes that inject trainable low-rank decomposition (A and B matrices) into frozen base layers. LoraLayer is the base class managing adapter dictionaries (lora_A, lora_B, scaling, dropout) and lifecycle methods. Linear, Embedding, Conv1d/2d/3d, and MultiheadAttention classes extend this for specific layer types. Each implements forward() to compute base_output + (LoRA_B @ LoRA_A @ dropout(input)) * scaling. Supports multiple adapters per layer, merge/unmerge operations, DoRA variants, and various initialization strategies (Kaiming, Gaussian, PiSSA, OLoRA, LoftQ, CorDA). dispatch_default() selects appropriate LoRA class based on base layer type.

**Significance:** The computational heart of LoRA in PEFT. These classes perform the actual low-rank adaptation during training and inference. By keeping base weights frozen and only training small A and B matrices, they enable parameter-efficient fine-tuning. Support for multiple layer types makes LoRA applicable to diverse architectures (transformers, CNNs, embeddings).
