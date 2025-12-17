# File: `src/peft/tuners/lora/variants.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 926 |
| Classes | `ArrowLinearVariant`, `DoraLinearVariant`, `DoraEmbeddingVariant`, `_DoraConvNdVariant`, `DoraConv1dVariant`, `DoraConv2dVariant`, `DoraConv3dVariant`, `QALoraLinearVariant`, `ALoraLinearVariant`, `BlockDiagonalLinear`, `BdLoraLinearVariant` |
| Functions | `calculate_alora_offsets`, `is_alora_relevant_in_batch`, `get_alora_offsets_for_forward`, `get_alora_offsets_for_generate` |
| Imports | __future__, accelerate, arrow, collections, config, dora, layer, peft, torch, typing, ... +1 more |

## Understanding

**Status:** ✅ Documented

**Purpose:** Advanced LoRA variant implementations extending base LoRA functionality

**Mechanism:** Implements LoraVariant protocol for specialized adapters: ArrowLinearVariant (mixture-of-experts routing), DoraLinearVariant/DoraEmbeddingVariant/DoraConvVariants (weight-decomposed adaptation), QALoraLinearVariant (quantization-aware with activation pooling), ALoraLinearVariant (activated LoRA with token-level triggering via invocation sequences), and BdLoraLinearVariant (block-diagonal structure). Each variant provides custom init/forward/merge/unmerge implementations.

**Significance:** Houses cutting-edge LoRA research implementations in production code. Provides users access to state-of-the-art techniques like DoRA's magnitude-direction decomposition, Arrow's adaptive routing, QALoRA's efficient quantized training, aLoRA's conditional activation, and block-diagonal structure for parameter efficiency. This file turns research papers into usable features.
