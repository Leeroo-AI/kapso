# File: `unsloth/kernels/cross_entropy_loss.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 459 |
| Classes | `Fast_CrossEntropyLoss` |
| Functions | `fast_cross_entropy_loss`, `patch_loss_functions` |
| Imports | packaging, torch, transformers, triton, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Optimized cross-entropy loss computation for language models

**Mechanism:** Implements custom Triton kernels for numerically stable cross-entropy calculation using logsumexp technique. Handles both regular vocabularies (<=65536) and large vocabularies (>65536) through chunking. Supports logit softcapping (for Gemma) and logit scaling (for Cohere). Provides forward and backward passes with gradients computed in-place for memory efficiency.

**Significance:** Critical optimization for LLM training - cross-entropy is computed at every training step. This kernel fuses operations to reduce memory bandwidth and supports model-specific features. The `patch_loss_functions` enables drop-in replacement of standard PyTorch/Transformers loss functions.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
