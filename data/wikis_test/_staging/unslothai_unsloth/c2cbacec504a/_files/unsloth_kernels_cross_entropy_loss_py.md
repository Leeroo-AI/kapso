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

**Purpose:** Implements highly optimized cross-entropy loss computation using Triton kernels with support for logit scaling and softcapping for models like Cohere and Gemma 2.

**Mechanism:** Two-stage Triton kernels (_cross_entropy_forward, _cross_entropy_backward) with chunked processing for large vocabularies. Forward pass uses stable logsumexp computation with optional logit transformations. Backward pass implements chain rule derivatives for gradient computation. Chunking strategy handles vocabularies > 65K by splitting into 65K blocks and performing hierarchical logsumexp reductions.

**Significance:** Dramatically accelerates loss computation during training (2x+ speedup typical), crucial for memory efficiency with large vocabularies and critical for accurate gradient computation with various normalization techniques.
