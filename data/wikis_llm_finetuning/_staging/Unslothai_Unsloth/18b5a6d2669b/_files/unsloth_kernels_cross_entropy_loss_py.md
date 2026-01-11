# File: `unsloth/kernels/cross_entropy_loss.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 459 |
| Classes | `Fast_CrossEntropyLoss` |
| Functions | `fast_cross_entropy_loss`, `patch_loss_functions` |
| Imports | torch, transformers, triton, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Triton optimized cross entropy loss with logit softcapping and chunked vocabulary support

**Mechanism:** Implements custom autograd function with Triton kernels for forward/backward passes. Handles small vocabs (<=65K) in single pass, large vocabs (>65K like Gemma 256K) via chunked logsumexp. Supports logit softcapping (Gemma 2: t*tanh(x/t)) and logit scaling (Cohere). Uses numerically stable logsumexp algorithm

**Significance:** Critical for training stability and speed. Enables efficient handling of models with massive vocabularies while supporting advanced features like softcapping. Faster than PyTorch default cross entropy and can be patched into transformers loss functions
