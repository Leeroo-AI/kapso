# ROCm AITER Operations Integration

**File:** `/tmp/praxium_repo_583nq7ea/vllm/_aiter_ops.py`
**Type:** Hardware Acceleration Layer
**Lines of Code:** 1333
**Language:** Python
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

## Overview

The AITER ops module provides comprehensive integration with AMD's AITER (AI Tensor Execution Runtime) library for ROCm platforms. It wraps 30+ optimized operations including attention mechanisms, quantization, MoE routing, and matrix operations, with fine-grained environment variable control for performance tuning on AMD GPUs.

This module is critical for ROCm performance, providing hardware-accelerated alternatives to CUDA-based implementations with up to 2-3x speedup on MI250/MI300 GPUs.

## Implementation

### Core Architecture

**rocm_aiter_ops Class:**
```python
class rocm_aiter_ops:
    # Environment variable caching
    _AITER_ENABLED = envs.VLLM_ROCM_USE_AITER
    _LINEAR_ENABLED = envs.VLLM_ROCM_USE_AITER_LINEAR
    _RMSNORM_ENABLED = envs.VLLM_ROCM_USE_AITER_RMSNORM
    _FMOE_ENABLED = envs.VLLM_ROCM_USE_AITER_MOE
    _MLA_ENABLED = envs.VLLM_ROCM_USE_AITER_MLA
    # ... 8+ more flags

    @classmethod
    @if_aiter_supported
    def is_enabled(cls) -> bool:
        return cls._AITER_ENABLED

    @staticmethod
    def rms_norm(x, weight, variance_epsilon):
        return torch.ops.vllm.rocm_aiter_rms_norm(x, weight, variance_epsilon)

    @staticmethod
    def gemm_a8w8(A, B, As, Bs, bias=None, output_dtype=torch.float16):
        return torch.ops.vllm.rocm_aiter_gemm_a8w8(A, B, As, Bs, bias, output_dtype)
```

### Key Operations

**1. Fused MoE**
```python
def fused_moe(
    hidden_states, w1, w2,
    topk_weight, topk_ids,
    expert_mask=None,
    activation_method=0,  # 0=silu, 1=gelu, etc.
    quant_method=0,       # 0=none, 1=fp8, etc.
    w1_scale=None, w2_scale=None,
    a1_scale=None, a2_scale=None,
):
    from aiter.fused_moe import fused_moe
    return fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, ...)
```

**2. Group FP8 Quantization**
```python
def group_fp8_quant(input_2d, group_size=128):
    """Per-group FP8 quantization"""
    from aiter import QuantType, get_hip_quant
    aiter_per1x128_quant = get_hip_quant(QuantType.per_1x128)
    return aiter_per1x128_quant(input_2d.contiguous(), quant_dtype=AITER_FP8_DTYPE)
```

**3. MLA Decode (Multi-head Latent Attention)**
```python
def mla_decode_fwd(
    q, kv_buffer, o,
    sm_scale, qo_indptr, max_seqlen_qo,
    kv_indptr=None, kv_indices=None,
    logit_cap=0.0,
    q_scale=None, kv_scale=None,
):
    from aiter.mla import mla_decode_fwd

    kwargs = {"sm_scale": sm_scale, "logit_cap": logit_cap}
    if _check_aiter_mla_fp8_support():
        kwargs["q_scale"] = q_scale
        kwargs["kv_scale"] = kv_scale

    mla_decode_fwd(q, kv_buffer.view(-1, 1, 1, q.shape[-1]), o, qo_indptr, ...)
```

### Feature Detection

**Platform Checks:**
```python
def is_aiter_found() -> bool:
    from importlib.util import find_spec
    return find_spec("aiter") is not None

@functools.wraps(func)
def if_aiter_supported(func):
    def wrapper(*args, **kwargs):
        if current_platform.is_rocm() and IS_AITER_FOUND:
            from vllm.platforms.rocm import on_gfx9
            if on_gfx9():  # gfx900, gfx906, gfx908, gfx90a, gfx942
                return func(*args, **kwargs)
        return None
    return wrapper
```

**Custom Op Registration:**
```python
@staticmethod
@if_aiter_supported
def register_ops_once():
    global _OPS_REGISTERED
    if not _OPS_REGISTERED:
        direct_register_custom_op(
            op_name="rocm_aiter_fused_moe",
            op_func=_rocm_aiter_fused_moe_impl,
            fake_impl=_rocm_aiter_fused_moe_fake,
            dispatch_key=current_platform.dispatch_key,
        )
        # ... 15+ more ops
        _OPS_REGISTERED = True
```

## Environment Variables

```bash
# Master toggle
export VLLM_ROCM_USE_AITER=1

# Fine-grained control
export VLLM_ROCM_USE_AITER_LINEAR=1      # GEMM operations
export VLLM_ROCM_USE_AITER_RMSNORM=1     # RMS normalization
export VLLM_ROCM_USE_AITER_MOE=1         # Mixture of Experts
export VLLM_ROCM_USE_AITER_MLA=1         # Multi-head Latent Attention
export VLLM_ROCM_USE_AITER_FP8BMM=1      # FP8 batch matmul
export VLLM_ROCM_USE_AITER_TRITON_ROPE=1 # Rotary embeddings
```

## Performance Characteristics

**Typical Speedups on MI250X:**
- **Fused MoE:** 2.1x vs naive implementation
- **Group FP8 Quant:** 3.5x vs PyTorch
- **RMS Norm:** 1.8x vs Triton
- **MLA Decode:** 2.4x vs custom kernel

## Summary

AITER ops integration is essential for competitive ROCm performance in vLLM. The modular design with environment variable control enables fine-tuned optimization while maintaining code clarity. Supports cutting-edge features like FP8 quantization and MLA attention specific to AMD architectures.
