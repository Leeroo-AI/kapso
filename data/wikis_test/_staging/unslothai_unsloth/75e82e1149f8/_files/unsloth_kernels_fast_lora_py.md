# File: `unsloth/kernels/fast_lora.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 717 |
| Classes | `LoRA_MLP`, `LoRA_QKV`, `LoRA_W` |
| Functions | `apply_lora_mlp_swiglu`, `apply_lora_mlp_geglu_exact`, `apply_lora_mlp_geglu_approx`, `apply_lora_qkv`, `apply_lora_o`, `fast_lora_forward` |
| Imports | geglu, swiglu, torch, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Fused LoRA (Low-Rank Adaptation) operations for efficient parameter-efficient fine-tuning.

**Mechanism:** Implements custom autograd functions that fuse LoRA adapter computations with base model weights. The key insight is computing W_out = (W + A@B) @ X by combining base weight matmul with low-rank adapter matmul in a single kernel. Handles three main cases: MLP layers (gate/up/down projections with activation functions), QKV projections (query/key/value), and output projections. Supports both quantized (4-bit, FP8) and unquantized base weights. The backward pass efficiently computes gradients for LoRA parameters using chain rule and matrix factorization properties.

**Significance:** LoRA is essential for memory-efficient fine-tuning of large language models. This fused implementation avoids materializing full-precision weights in memory and reduces the number of GPU kernel launches. By combining activation functions (SwiGLU/GeGLU) with LoRA computations, it achieves substantial speedups over standard PEFT implementations. The support for various quantization formats makes it practical for training very large models on consumer hardware.
