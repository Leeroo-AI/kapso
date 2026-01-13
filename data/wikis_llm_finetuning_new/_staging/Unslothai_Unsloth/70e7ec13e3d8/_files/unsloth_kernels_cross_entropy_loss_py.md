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

**Purpose:** Implements a fast, memory-efficient cross-entropy loss using Triton GPU kernels with support for large vocabularies, logit softcapping, and logit scaling.

**Mechanism:** Provides Triton JIT-compiled kernels for both forward and backward passes. The forward pass computes logsumexp with numerical stability (using max subtraction trick) and handles vocabularies larger than 65536 through chunked computation that divides the vocab into blocks and combines partial logsumexp results. Supports logit softcapping (used by Gemma 2: t * tanh(x/t)) and logit scaling (used by Cohere: s * x). The backward pass computes gradients as softmax - one_hot(label), properly handling the softcapping/scaling chain rule. The Fast_CrossEntropyLoss class wraps these kernels in PyTorch's autograd.Function for automatic differentiation. Also provides patch_loss_functions() to replace standard transformers loss computation.

**Significance:** Core training optimization component. Cross-entropy is computed on every training step, so fusing the computation into optimized Triton kernels significantly reduces memory usage and improves throughput compared to PyTorch's default implementation.
