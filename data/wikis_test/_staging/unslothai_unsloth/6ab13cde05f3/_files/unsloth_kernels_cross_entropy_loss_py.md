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

**Purpose:** Optimized cross-entropy loss computation

**Mechanism:** Implements Triton kernels for fast cross-entropy loss calculation in both forward and backward passes. Uses numerically-stable logsumexp for preventing numerical overflow. Handles large vocabularies (>65K) by chunking the computation into manageable blocks, then reducing via logsumexp of chunks. Supports logit scaling (Cohere) and softcapping (Gemma 2).

**Significance:** Significantly accelerates loss computation during training by leveraging GPU parallelism through Triton. Enables faster training iterations, especially critical for models with large vocabularies. Includes support for modern model variants with special loss transformations.
