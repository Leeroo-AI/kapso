{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::GPU_Operations]], [[domain::ROCm]], [[domain::AMD_Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
ROCm/AMD GPU-specific optimizations through the AITER library for high-performance inference on Radeon GPUs.

=== Description ===
The _aiter_ops.py module provides a comprehensive interface to the AITER (AMD Inference Tensor Engine Runtime) library, which contains highly optimized kernels specifically tuned for AMD ROCm GPUs. This 1339-line module serves as a bridge between vLLM's architecture-agnostic API and AMD's specialized implementations for GFX9 architecture (MI200/MI300 series accelerators).

The module includes optimizations for key operations: (1) Fused MoE (Mixture of Experts) operations with FP8/INT8 quantization support and various activation functions; (2) Grouped TopK operations for efficient expert routing in MoE layers; (3) Multi-Latent Attention (MLA) decode operations for models like DeepSeek-V3; (4) FP8 GEMM operations with block-scale quantization; (5) RMSNorm and fused normalization operations; (6) Triton-based custom kernels for specific patterns.

The module uses a decorator pattern (if_aiter_supported) to gracefully handle cases where AITER is not available or the GPU architecture is unsupported. It provides both implementation and fake (shape-only) versions of all operations for compatibility with torch.compile. Environment variables control which AITER optimizations are enabled, allowing fine-grained performance tuning.

=== Usage ===
Automatically used by vLLM when running on supported ROCm platforms with AITER installed. Not typically called directly by users.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py vllm/_aiter_ops.py]
* '''Lines:''' 1-1339

=== Signature ===
<syntaxhighlight lang="python">
# AITER availability checking
def is_aiter_found() -> bool
def is_aiter_found_and_supported() -> bool
def if_aiter_supported(func: Callable) -> Callable

# Fused MoE Operations
def _rocm_aiter_fused_moe_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: torch.Tensor | None = None,
    activation_method: int = 0,
    quant_method: int = 0,
    doweight_stage1: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
) -> torch.Tensor

# Grouped TopK Operations
def _rocm_aiter_grouped_topk_impl(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
) -> None

# Multi-Latent Attention
def _rocm_aiter_mla_decode_fwd_impl(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    max_seqlen_qo: int,
    kv_indptr: torch.Tensor | None = None,
    kv_indices: torch.Tensor | None = None,
    kv_last_page_lens: torch.Tensor | None = None,
    sm_scale: float = 1.0,
    logit_cap: float = 0.0,
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
) -> None

# FP8 GEMM Operations
def _rocm_aiter_gemm_a8w8_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm._aiter_ops import (
    is_aiter_found_and_supported,
    if_aiter_supported,
)

# Check if AITER is available
if is_aiter_found_and_supported():
    print("AITER optimizations available on this ROCm GPU")
</syntaxhighlight>

== I/O Contract ==

=== Key Exports ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| is_aiter_found || Function || Check if AITER package is installed
|-
| is_aiter_found_and_supported || Function || Check if AITER is available and GPU is supported
|-
| if_aiter_supported || Decorator || Only execute function if AITER is supported
|-
| _rocm_aiter_fused_moe_impl || Function || Optimized fused MoE computation
|-
| _rocm_aiter_grouped_topk_impl || Function || Grouped TopK for MoE routing
|-
| _rocm_aiter_topk_softmax_impl || Function || TopK with softmax normalization
|-
| _rocm_aiter_mla_decode_fwd_impl || Function || Multi-latent attention decode
|-
| _rocm_aiter_gemm_a8w8_impl || Function || FP8 matrix multiplication
|-
| _rocm_aiter_gemm_a8w8_blockscale_impl || Function || FP8 GEMM with block scaling
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from vllm._aiter_ops import is_aiter_found_and_supported

# Example 1: Check AITER availability
if is_aiter_found_and_supported():
    print("Running on AMD GPU with AITER support")
    import vllm.envs as envs

    # Check which AITER features are enabled
    print(f"AITER MOE: {envs.VLLM_ROCM_USE_AITER_MOE}")
    print(f"AITER MLA: {envs.VLLM_ROCM_USE_AITER_MLA}")
    print(f"AITER Linear: {envs.VLLM_ROCM_USE_AITER_LINEAR}")
    print(f"AITER RMSNorm: {envs.VLLM_ROCM_USE_AITER_RMSNORM}")

# Example 2: Using the decorator pattern
from vllm._aiter_ops import if_aiter_supported

@if_aiter_supported
def my_rocm_optimized_function():
    """This function only runs on supported ROCm hardware with AITER"""
    print("Using AITER optimizations")
    # Perform AITER-accelerated operations
    return True

result = my_rocm_optimized_function()
# result will be None if AITER is not supported

# Example 3: Accessing AITER operations (internal usage)
# These are typically called by vLLM's layer implementations
from vllm._aiter_ops import _rocm_aiter_fused_moe_impl

if is_aiter_found_and_supported():
    # FP16 MoE example
    hidden_states = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
    w1 = torch.randn(8, 11008, 4096, dtype=torch.float16, device="cuda")  # 8 experts
    w2 = torch.randn(8, 4096, 11008, dtype=torch.float16, device="cuda")
    topk_weights = torch.randn(1024, 2, device="cuda")
    topk_ids = torch.randint(0, 8, (1024, 2), device="cuda")

    # Call AITER fused MoE
    output = _rocm_aiter_fused_moe_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        activation_method=1,  # SiLU
        quant_method=0,  # FP16
    )
</syntaxhighlight>

== Related Pages ==
* [[implements::Interface:vllm-project_vllm_Platform_Specific_Ops]]
* [[requires::Library:AITER]]
* [[uses::Module:vllm-project_vllm_Custom_Ops]]
* [[optimizes::Hardware:AMD_ROCm_GPUs]]
* [[related::Module:vllm-project_vllm_Environment_Variables]]
