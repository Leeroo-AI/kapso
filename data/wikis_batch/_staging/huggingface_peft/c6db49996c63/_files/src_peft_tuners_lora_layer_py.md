# File: `src/peft/tuners/lora/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 2304 |
| Classes | `LoraVariant`, `LoraLayer`, `Linear`, `Embedding`, `_ConvNd`, `Conv2d`, `Conv1d`, `Conv3d`, `MultiheadAttention`, `_LoraParameterProxy`, `ParamWrapper` |
| Functions | `dispatch_default` |
| Imports | __future__, config, contextlib, math, peft, torch, transformers, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Core LoRA layer implementations for all standard PyTorch layer types

**Mechanism:** Defines LoraLayer base class and concrete implementations for Linear, Embedding, Conv1d/2d/3d, and MultiheadAttention. Each layer maintains lora_A (down-projection) and lora_B (up-projection) matrices, handles multiple adapters via ModuleDict, implements merge/unmerge operations, and supports variants through the LoraVariant protocol. Includes ParamWrapper for targeting nn.Parameter directly.

**Significance:** Foundational module providing the actual LoRA adapter implementations. Contains 2300+ lines of carefully crafted logic for weight updates (W = W0 + BA*scale), adapter activation/deactivation, initialization methods (random, Kaiming, PiSSA, OLoRA, LoftQ, CorDA, EVA), dropout, mixed-precision handling, and variant dispatching. Every LoRA operation ultimately uses these layer classes.
