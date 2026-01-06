{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::GPU_Operations]], [[domain::Kernels]], [[domain::Performance]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
High-performance custom CUDA/ROCm kernel operations for optimized transformer model inference.

=== Description ===
The _custom_ops.py module provides a comprehensive collection of custom GPU kernels and operations that are critical for vLLM's performance. This 3116-line module wraps highly optimized C++/CUDA implementations for core transformer operations including paged attention, rotary embeddings, layer normalization, activation functions, and various quantization operations.

Key operation categories include: (1) Paged attention operations (v1, v2, ROCm variants) that enable efficient KV cache management with block-based memory allocation; (2) Position encoding operations like rotary embeddings that apply positional information efficiently; (3) Layer normalization variants including RMS norm and fused add+RMS norm for reduced memory bandwidth; (4) Activation functions (SiLU, GELU variants) often fused with gating mechanisms; (5) Quantization operations supporting FP8, INT8, INT4, and various mixed-precision formats; (6) Cache management operations for reshaping and storing KV caches.

The module intelligently selects between CUDA and ROCm implementations based on the platform, with specialized paths for CPU operations when needed. It also provides fake implementations for torch.compile compatibility and abstract shape inference. These custom operations are fundamental to vLLM achieving state-of-the-art inference throughput.

=== Usage ===
Used internally by vLLM attention layers, linear layers, and model implementations. Not typically called directly by users.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py vllm/_custom_ops.py]
* '''Lines:''' 1-3116

=== Signature ===
<syntaxhighlight lang="python">
# Paged Attention Operations
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None

# Position Encoding
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None

# Layer Normalization
def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float
) -> None

def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float
) -> None

# Activation Functions
def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None
def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None

# Cache Management
def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm._custom_ops import (
    paged_attention_v1,
    rotary_embedding,
    rms_norm,
    fused_add_rms_norm,
    silu_and_mul,
)
</syntaxhighlight>

== I/O Contract ==

=== Key Exports ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| paged_attention_v1 || Function || Single-query paged attention kernel
|-
| paged_attention_v2 || Function || Multi-query paged attention kernel with partitioning
|-
| paged_attention_rocm || Function || ROCm-optimized paged attention variant
|-
| rotary_embedding || Function || Apply rotary position embeddings
|-
| rms_norm || Function || Root mean square layer normalization
|-
| fused_add_rms_norm || Function || Fused residual add + RMS norm
|-
| silu_and_mul || Function || SiLU activation + gating multiplication
|-
| gelu_and_mul || Function || GELU activation + gating multiplication
|-
| reshape_and_cache || Function || Reshape KV tensors and store in paged cache
|-
| apply_repetition_penalties || Function || Apply repetition penalties to logits
|-
| merge_attn_states || Function || Merge attention states from prefix/suffix
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from vllm._custom_ops import paged_attention_v1, rms_norm, rotary_embedding

# Example 1: Paged Attention
batch_size, num_heads, head_size = 4, 32, 128
seq_len, num_kv_heads = 100, 8
block_size, max_seq_len = 16, 2048

query = torch.randn(batch_size, num_heads, head_size, device="cuda")
key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, device="cuda")
value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, device="cuda")
out = torch.empty_like(query)
block_tables = torch.randint(0, num_blocks, (batch_size, max_blocks), device="cuda")
seq_lens = torch.randint(1, max_seq_len, (batch_size,), device="cuda")

paged_attention_v1(
    out=out,
    query=query,
    key_cache=key_cache,
    value_cache=value_cache,
    num_kv_heads=num_kv_heads,
    scale=1.0 / (head_size ** 0.5),
    block_tables=block_tables,
    seq_lens=seq_lens,
    block_size=block_size,
    max_seq_len=max_seq_len,
    alibi_slopes=None,
    kv_cache_dtype="auto",
    k_scale=torch.tensor(1.0),
    v_scale=torch.tensor(1.0),
)

# Example 2: RMS Normalization
hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")
weight = torch.randn(hidden_dim, device="cuda")
out = torch.empty_like(hidden_states)

rms_norm(out, hidden_states, weight, epsilon=1e-6)

# Example 3: Rotary Embeddings
positions = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)
query = torch.randn(batch_size, seq_len, num_heads * head_size, device="cuda")
key = torch.randn(batch_size, seq_len, num_kv_heads * head_size, device="cuda")
cos_sin_cache = torch.randn(max_seq_len, head_size, device="cuda")

rotary_embedding(
    positions=positions,
    query=query,
    key=key,
    head_size=head_size,
    cos_sin_cache=cos_sin_cache,
    is_neox=True,
)
</syntaxhighlight>

== Related Pages ==
* [[uses::Module:vllm-project_vllm_Paged_Attention]]
* [[implements::Interface:vllm-project_vllm_Attention_Backend]]
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related::Module:vllm-project_vllm_AITER_Ops]]
