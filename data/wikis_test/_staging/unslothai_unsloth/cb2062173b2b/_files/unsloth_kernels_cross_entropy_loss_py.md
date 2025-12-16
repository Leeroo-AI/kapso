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

**Purpose:** Provides optimized Triton kernels for computing cross-entropy loss with support for large vocabularies (256K+), logit softcapping (Gemma 2), and logit scaling (Cohere). Significantly faster than PyTorch's default implementation.

**Mechanism:** Implements custom forward/backward passes using Triton kernels. For vocabs ≤65536 (Llama, Mistral), uses single-pass `_cross_entropy_forward`. For larger vocabs (Gemma 256K), uses `_chunked_cross_entropy_forward` that splits computation across chunks and combines logsumexp results. Computes stable logsumexp using `c = max(x)` trick to prevent overflow. Backward pass efficiently computes gradients as `exp(x - logsumexp) - 1` for labels and `exp(x - logsumexp)` for non-labels. Handles label masking (-100) and optional softcapping/scaling.

**Significance:** Critical performance optimization for training. Cross-entropy loss is computed after every forward pass, so optimizing it provides substantial speedups. The chunked implementation enables training with very large vocabularies that would otherwise exceed GPU memory limits. Support for softcapping and scaling enables compatibility with modern architectures like Gemma 2 and Cohere.
