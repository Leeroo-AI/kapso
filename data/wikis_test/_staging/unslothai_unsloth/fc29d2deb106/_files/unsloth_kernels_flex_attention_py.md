# File: `unsloth/kernels/flex_attention.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 187 |
| Functions | `slow_inference_attention_softcapping` |
| Imports | functools, os, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Flexible attention mechanisms with logit softcapping support

**Mechanism:** Provides two implementations: (1) torch.compile-optimized attention with tanh-based softcapping for models like Gemma, handling grouped query attention expansion, and (2) PyTorch 2.5+ flex_attention API with block masks for causal and sliding window patterns. Falls back to simpler inference attention without torch.compile for better compatibility.

**Significance:** Enables efficient attention variants required by newer models (Gemma's softcapping, sliding window attention). The flex_attention path uses PyTorch's latest optimization features when available, while maintaining backward compatibility. Essential for supporting diverse attention patterns across model architectures.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
