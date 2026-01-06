# File: `src/peft/tuners/lora/dora.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 203 |
| Classes | `DoraLinearLayer`, `DoraEmbeddingLayer`, `_DoraConvNdLayer`, `DoraConv1dLayer`, `DoraConv2dLayer`, `DoraConv3dLayer` |
| Imports | copy, peft, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** DoRA helper functions

**Mechanism:** Provides utility functions for DoRA (Weight-Decomposed Low-Rank Adaptation). dora_layer_norm_before_merging() computes layer normalization on combined base+LoRA weights before merging. magnitude_vector_scaling() calculates the magnitude scaling factor that makes DoRA work. copy_dora_attr_from_another_adapter() duplicates DoRA parameters when copying adapters. convert_dora_to_lora() transforms DoRA adapters back to vanilla LoRA by merging magnitude vectors into the LoRA weights, useful for deployment.

**Significance:** Essential infrastructure for DoRA functionality. DoRA improves upon standard LoRA by decomposing weight updates into magnitude and direction components, leading to better training dynamics and convergence. These utilities enable DoRA's weight normalization approach while maintaining compatibility with the standard LoRA interface.
