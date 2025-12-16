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

**Purpose:** Efficient LoRA forward and backward passes

**Mechanism:** Implements fused LoRA matrix multiplications for MLP and attention layers. The LoRA_MLP class combines gate, up, and down projections into single autograd functions with custom forward/backward kernels. LoRA_QKV and LoRA_W handle query-key-value and single-weight projections. Uses quantization-aware operations for 4-bit and FP8 models. Supports both SwiGLU and GeGLU activation types.

**Significance:** Dramatically reduces memory footprint and computation time for LoRA fine-tuning by fusing multiple operations. Eliminates intermediate tensor materialization and enables in-place gradient computation. Supports quantized models, making LoRA-enhanced inference/training more efficient on memory-constrained hardware.
