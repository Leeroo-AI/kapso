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

**Purpose:** Optimized cross-entropy loss computation using Triton kernels for faster training.

**Mechanism:** Implements custom forward and backward passes using Triton JIT-compiled kernels. The forward pass computes logsumexp in a numerically stable manner and handles large vocabularies (>65K) via chunking. Supports logit softcapping (for Gemma 2) and logit scaling (for Cohere). The backward pass computes gradients using the chain rule with exp(x - logsumexp) for softmax derivatives. Includes specialized kernels for both standard (<65K vocab) and chunked (>65K vocab) computation.

**Significance:** Cross-entropy loss is the most computationally expensive part of language model training. This optimized implementation provides significant speedups by fusing operations in GPU kernels, reducing memory transfers, and handling special cases like padding tokens (-100) efficiently. The chunking strategy enables support for models with extremely large vocabularies like Gemma 256K.
