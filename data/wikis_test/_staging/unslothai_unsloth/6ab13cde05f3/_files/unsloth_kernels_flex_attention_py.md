# File: `unsloth/kernels/flex_attention.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 187 |
| Functions | `slow_inference_attention_softcapping` |
| Imports | functools, os, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Flexible attention with logit softcapping

**Mechanism:** Provides attention mechanisms with optional logit softcapping (used in Gemma models). When flex_attention is available (PyTorch 2.5+), uses compiled flex_attention from torch.nn with custom score modifications. Falls back to manual implementation combining Q@K.T, softcapping via tanh, and softmax. Includes utilities for creating causal masks and sliding window masks compatible with flex_attention.

**Significance:** Enables efficient softcapped attention for advanced models like Gemma while maintaining backward compatibility. Leverages PyTorch's optimized attention kernels when available, otherwise provides a compiled fallback implementation. Supports both training and inference modes with proper mask handling.
