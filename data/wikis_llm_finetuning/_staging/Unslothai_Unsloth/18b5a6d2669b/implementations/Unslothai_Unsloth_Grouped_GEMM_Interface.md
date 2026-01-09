# Implementation: Grouped_GEMM_Interface

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Kernels]], [[domain::MoE]], [[domain::GEMM]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
High-level interface for grouped GEMM operations optimized for Mixture-of-Experts (MoE) model MLPs with TMA support.

=== Description ===
This module provides the main entry points for grouped GEMM operations used in MoE architectures. It supports various fusions (permute_x, permute_y, fuse_mul_post) specific to MoE workflows, autotuning, and Tensor Memory Accelerator (TMA) on SM90+ GPUs. The interface abstracts over forward, dW (weight gradient), and dX (input gradient) kernels.

=== Usage ===
Use these functions when implementing MoE layers or when you need batched matrix multiplications with variable-size groups. The grouped_gemm_forward handles the expert dispatch pattern efficiently.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/moe/grouped_gemm/interface.py unsloth/kernels/moe/grouped_gemm/interface.py]
* '''Lines:''' 1-968

=== Key Functions ===
<syntaxhighlight lang="python">
def grouped_gemm_forward(
    X: torch.Tensor,              # Input activations
    W: torch.Tensor,              # Expert weights [E, N, K]
    topk: int,                    # Number of experts per token
    m_sizes: torch.Tensor,        # Tokens per expert
    gather_indices: torch.Tensor = None,  # Token-to-expert indices
    topk_weights: torch.Tensor = None,    # Expert routing weights
    # Fusions
    permute_x: bool = False,      # Fuse input permutation
    permute_y: bool = False,      # Fuse output permutation
    fuse_mul_post: bool = False,  # Fuse routing weight multiplication
    # Autotuning
    autotune: bool = False,
    # Manual kernel params
    BLOCK_SIZE_M: int = 32,
    BLOCK_SIZE_N: int = 32,
    BLOCK_SIZE_K: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    # TMA options (SM90+)
    use_tma_load_w: bool = False,
    use_tma_load_x: bool = False,
    use_tma_store: bool = False,
) -> torch.Tensor:
    """
    Grouped GEMM forward pass for MoE MLPs.

    Returns:
        y: (total_tokens, N) output of grouped GEMM
    """

def supports_tma() -> bool:
    """Check if GPU supports Tensor Memory Accelerator (SM90+)."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.moe.grouped_gemm.interface import (
    grouped_gemm_forward,
    supports_tma,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| X || Tensor || Yes || Input [num_tokens, K] or [total_tokens, K]
|-
| W || Tensor || Yes || Expert weights [E, N, K]
|-
| topk || int || Yes || Number of experts per token
|-
| m_sizes || Tensor || Yes || Number of tokens per expert [E]
|-
| gather_indices || Tensor || When permute_x/y || Token indices [total_tokens]
|-
| topk_weights || Tensor || When fuse_mul_post || Routing weights [total_tokens]
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| y || Tensor || Output [total_tokens, N] or [num_tokens, N] if permute_y
|}

== Fusion Options ==

{| class="wikitable"
|-
! Option !! Description !! Use Case
|-
| permute_x || Fuse token-to-expert permutation || First GEMM in MoE MLP
|-
| permute_y || Fuse expert-to-token unpermutation || Second GEMM in MoE MLP
|-
| fuse_mul_post || Multiply output by routing weights || Inference only (not training)
|}

== Usage Examples ==

=== Basic MoE Forward ===
<syntaxhighlight lang="python">
from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm_forward
import torch

# Example MoE configuration
num_experts = 8
topk = 2
hidden_dim = 4096
intermediate_dim = 14336
num_tokens = 512

# Expert weights
W = torch.randn(num_experts, intermediate_dim, hidden_dim,
                device="cuda", dtype=torch.bfloat16)

# Input activations (already permuted to expert order)
X = torch.randn(num_tokens * topk, hidden_dim,
                device="cuda", dtype=torch.bfloat16)

# Tokens per expert
m_sizes = torch.tensor([128, 128, 128, 128, 128, 128, 128, 128],
                       device="cuda", dtype=torch.int32)

# Run grouped GEMM
output = grouped_gemm_forward(
    X=X, W=W, topk=topk, m_sizes=m_sizes,
    BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
    num_warps=4, num_stages=2,
)
# output shape: [1024, 14336]
</syntaxhighlight>

=== With Input Permutation Fusion ===
<syntaxhighlight lang="python">
# Input in token order (not yet sorted by expert)
X_token_order = torch.randn(num_tokens, hidden_dim,
                            device="cuda", dtype=torch.bfloat16)

# Token-to-expert assignment indices
gather_indices = torch.randint(0, num_tokens, (num_tokens * topk,),
                               device="cuda", dtype=torch.int32)

output = grouped_gemm_forward(
    X=X_token_order,
    W=W,
    topk=topk,
    m_sizes=m_sizes,
    gather_indices=gather_indices,
    permute_x=True,  # Fuse permutation
)
</syntaxhighlight>

=== With Autotuning ===
<syntaxhighlight lang="python">
output = grouped_gemm_forward(
    X=X, W=W, topk=topk, m_sizes=m_sizes,
    autotune=True,  # Let Triton find best kernel config
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
