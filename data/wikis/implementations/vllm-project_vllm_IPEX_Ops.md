{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::CPU_Operations]], [[domain::Intel_Optimization]], [[domain::XPU_Acceleration]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Intel CPU and XPU optimizations through the Intel Extension for PyTorch (IPEX) for efficient inference on Intel hardware.

=== Description ===
The _ipex_ops.py module provides a unified interface to Intel Extension for PyTorch (IPEX) optimized operations for both Intel CPUs and Intel XPU (Data Center GPU Max series) accelerators. This 457-line module wraps IPEX's specialized implementations of key transformer operations, enabling vLLM to achieve high performance on Intel hardware.

The module implements the same operation signatures as the CUDA/ROCm variants but routes them through IPEX-optimized paths. Key operations include: (1) Activation functions (SiLU, GELU variants) optimized for Intel vector instructions; (2) Paged attention v1/v2 using IPEX's PagedAttention implementation; (3) Rotary embeddings with efficient CPU/XPU implementations; (4) RMS normalization and fused operations; (5) Variable-length attention for prefill operations; (6) KV cache management (reshape_and_cache, reshape_and_cache_flash); (7) FlashAttention-style variable-length attention; (8) FP8 quantization support on XPU; (9) Block copy and swap operations for KV cache management.

The module automatically detects whether it's running on CPU or XPU builds of IPEX and selects appropriate implementations. It provides seamless integration with vLLM's architecture-agnostic layer implementations, allowing the same model code to run efficiently on Intel hardware without modifications.

=== Usage ===
Automatically used by vLLM when running on Intel CPUs or XPUs with IPEX installed. Operations are called through vLLM's platform abstraction layer.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/_ipex_ops.py vllm/_ipex_ops.py]
* '''Lines:''' 1-457

=== Signature ===
<syntaxhighlight lang="python">
class ipex_ops:
    @staticmethod
    def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None

    @staticmethod
    def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None

    @staticmethod
    def paged_attention_v1(
        out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        max_context_len: int,
        alibi_slopes: torch.Tensor | None,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> None

    @staticmethod
    def rotary_embedding(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
    ) -> None

    @staticmethod
    def rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor

    @staticmethod
    def varlen_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        seqlen_q: torch.Tensor,
        seqlen_k: torch.Tensor,
        alibi_slopes: torch.Tensor | None,
        max_seqlen_q: int,
        max_seqlen_k: int,
        pdropout: float,
        softmax_scale: float,
        zero_tensors: bool,
        is_causal: bool,
        return_softmax: bool,
        gen_: torch.Generator,
        window_size_left: float,
        window_size_right: float,
        logits_soft_cap: float,
    ) -> None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm._ipex_ops import ipex_ops

# Use IPEX operations through the ipex_ops class
out = torch.empty_like(input)
ipex_ops.silu_and_mul(out, input)
</syntaxhighlight>

== I/O Contract ==

=== Key Exports ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| ipex_ops || Class || Static class containing all IPEX operations
|-
| ipex_ops.silu_and_mul || Method || Fused SiLU activation and multiplication
|-
| ipex_ops.gelu_and_mul || Method || Fused GELU activation and multiplication
|-
| ipex_ops.paged_attention_v1 || Method || Single-query paged attention
|-
| ipex_ops.paged_attention_v2 || Method || Multi-query paged attention
|-
| ipex_ops.rotary_embedding || Method || Apply rotary position embeddings
|-
| ipex_ops.rms_norm || Method || RMS layer normalization
|-
| ipex_ops.fused_add_rms_norm || Method || Fused residual add + RMS norm
|-
| ipex_ops.varlen_attention || Method || Variable-length attention
|-
| ipex_ops.reshape_and_cache || Method || Store KV in paged cache
|-
| ipex_ops.flash_attn_varlen_func || Method || FlashAttention-style variable-length attention
|-
| ipex_ops.scaled_fp8_quant || Method || FP8 quantization (XPU only)
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from vllm._ipex_ops import ipex_ops

# Example 1: Activation functions
batch_size, hidden_dim = 32, 4096
x = torch.randn(batch_size, 2 * hidden_dim)
out = torch.empty(batch_size, hidden_dim)

# Fused SiLU and gating multiplication
ipex_ops.silu_and_mul(out, x)

# Example 2: RMS Normalization
hidden_states = torch.randn(batch_size, 128, hidden_dim)
weight = torch.randn(hidden_dim)
normalized = ipex_ops.rms_norm(hidden_states, weight, epsilon=1e-6)

# Example 3: Paged Attention on Intel XPU
num_heads, head_size = 32, 128
query = torch.randn(batch_size, num_heads, head_size, device="xpu")
key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, device="xpu")
value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, device="xpu")
out = torch.empty_like(query)

ipex_ops.paged_attention_v1(
    out=out,
    query=query,
    key_cache=key_cache,
    value_cache=value_cache,
    num_kv_heads=8,
    scale=1.0 / (head_size ** 0.5),
    block_tables=block_tables,
    context_lens=context_lens,
    block_size=16,
    max_context_len=2048,
    alibi_slopes=None,
    kv_cache_dtype="auto",
    k_scale=1.0,
    v_scale=1.0,
)

# Example 4: Rotary Embeddings
positions = torch.arange(seq_len, device="cpu").unsqueeze(0)
query = torch.randn(batch_size, seq_len, num_heads * head_size, device="cpu")
key = torch.randn(batch_size, seq_len, num_kv_heads * head_size, device="cpu")
cos_sin_cache = torch.randn(max_seq_len, head_size, device="cpu")

ipex_ops.rotary_embedding(
    positions=positions,
    query=query,
    key=key,
    head_size=head_size,
    cos_sin_cache=cos_sin_cache,
    is_neox=True,
)

# Example 5: Variable-length Attention (Prefill)
total_tokens = 1024
query = torch.randn(total_tokens, num_heads, head_size, device="xpu")
key = torch.randn(total_tokens, num_kv_heads, head_size, device="xpu")
value = torch.randn(total_tokens, num_kv_heads, head_size, device="xpu")
out = torch.empty_like(query)
cu_seqlens_q = torch.tensor([0, 256, 512, 768, 1024], dtype=torch.int32, device="xpu")
cu_seqlens_k = cu_seqlens_q.clone()

ipex_ops.varlen_attention(
    query=query,
    key=key,
    value=value,
    out=out,
    seqlen_q=cu_seqlens_q,
    seqlen_k=cu_seqlens_k,
    alibi_slopes=None,
    max_seqlen_q=256,
    max_seqlen_k=256,
    pdropout=0.0,
    softmax_scale=1.0 / (head_size ** 0.5),
    zero_tensors=False,
    is_causal=True,
    return_softmax=False,
    gen_=None,
    window_size_left=-1,
    window_size_right=-1,
    logits_soft_cap=0.0,
)
</syntaxhighlight>

== Related Pages ==
* [[implements::Interface:vllm-project_vllm_Platform_Specific_Ops]]
* [[requires::Library:Intel_Extension_for_PyTorch]]
* [[optimizes::Hardware:Intel_CPUs_and_XPUs]]
* [[related::Module:vllm-project_vllm_Custom_Ops]]
