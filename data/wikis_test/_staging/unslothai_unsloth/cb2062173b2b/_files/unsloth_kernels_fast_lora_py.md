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

**Purpose:** Implements fused LoRA (Low-Rank Adaptation) operations that combine base weight matrix multiplication with LoRA adapter computations in single kernels. Provides specialized implementations for MLP layers (SwiGLU/GEGLU), attention QKV projections, and output projections.

**Mechanism:** Three main autograd functions: (1) `LoRA_MLP` - fuses gate/up/down projections with LoRA adapters for SwiGLU/GEGLU MLPs, computing `h = activation(X@(G+Ag@Bg), X@(U+Au@Bu)) @ (W+Aw@Bw)` in single pass; (2) `LoRA_QKV` - computes Q, K, V projections with LoRA simultaneously `Q=X@(Wq+Aq@Bq), K=X@(Wk+Ak@Bk), V=X@(Wv+Av@Bv)`; (3) `LoRA_W` - generic single projection with LoRA. Backward passes efficiently compute gradients for both base weights and LoRA matrices using matmul_lora helper. Supports quantized base weights (4-bit, FP8) via dequantization.

**Significance:** Core to Unsloth's efficiency for LoRA finetuning. Instead of computing base layer and LoRA separately then adding (2 matmuls), these kernels fuse operations into 1 matmul equivalent. This is especially important for MLP layers where multiple projections can be fused. The quantization support enables efficient training of large models in limited memory while maintaining speed.
