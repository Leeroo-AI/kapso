# Implementation: MoE Ops

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::MoE]], [[domain::Model_Architecture]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

The MoE Ops module provides common utility functions for Mixture-of-Experts implementations. These operations form the foundation for token routing, permutation, and grouped matrix multiplication in MoE architectures. The module includes:

* '''permute''': Reorders tokens from token-order to expert-order for efficient grouped computation
* '''unpermute''': Restores tokens from expert-order back to original token-order
* '''calculate_topk''': Computes top-k expert selection with configurable activation (sigmoid/softmax)
* '''get_routing_indices''': Generates token counts and gather indices for expert routing
* '''torch_grouped_gemm''': Reference torch-native grouped GEMM implementation

These operations are used by model-specific MoE implementations (Llama4, Qwen3) and serve as reference implementations for validating optimized Triton kernels.

== Code Reference ==

'''File Path:''' `unsloth/kernels/moe/grouped_gemm/reference/moe_ops.py`

'''Function: permute'''

<syntaxhighlight lang="python">
def permute(X: torch.Tensor, gather_indices: torch.Tensor, topk: int):
    """
    Scatters X to a new tensor with shape [total_tokens, hidden_dim] where
    total_tokens is num_tokens * topk, permuting tokens according to sorted_token_idx.

    Helper for grouped gemm where hidden states need be ordered by expert.

    X: [num_tokens, hidden_dim]
    sorted_token_idx: [num_tokens * topk]
    topk: int

    Returns:
        [total_tokens, hidden_dim]
    """
    assert gather_indices.ndim == 1
    X = X.view(-1, X.shape[-1])
    # Shortcut for topk == 1
    if topk == 1:
        return X[gather_indices]
    return X[gather_indices // topk]
</syntaxhighlight>

'''Function: unpermute'''

<syntaxhighlight lang="python">
def unpermute(X: torch.Tensor, gather_indices: torch.Tensor):
    X = X.view(-1, X.shape[-1]) if X.ndim > 2 else X
    unpermuted = torch.empty_like(X)
    unpermuted.index_copy_(0, gather_indices, X)
    return unpermuted.view_as(X)
</syntaxhighlight>

'''Function: calculate_topk'''

<syntaxhighlight lang="python">
def calculate_topk(
    gating_output: torch.Tensor,
    top_k: int,
    use_sigmoid: bool,
    renormalize: bool,
    pre_act: bool = True,
    post_act: bool = False,
):
    """
    If post_act is True, then activation function is run AFTER topk
    If post_act is False, then activation function is run BEFORE topk

    This aligns with triton_bench implementation (post_act) whereas most models
    use pre_act (e.g. llama4, deepseek)
    """
</syntaxhighlight>

'''Function: get_routing_indices'''

<syntaxhighlight lang="python">
@torch.no_grad()
def get_routing_indices(
    selected_experts, num_experts, return_scatter_indices: bool = False
):
    """
    Returns:
        token_counts_by_expert: [num_experts]
        gather_indices: [num_tokens]
        scatter_indices [Optional]: Indices for unpermuting gathered inputs back
                                    to token order, shape (bs * seqlen * top_k,)
    """
    # Group tokens by expert indices using histogram
    token_counts_by_expert = torch.histc(
        selected_experts.view(-1),
        bins=num_experts,
        min=0,
        max=num_experts,
    )
    # Sort to get gather indices
    gather_indices = torch.argsort(selected_experts.view(-1), stable=True)

    if return_scatter_indices:
        scatter_indices = gather_indices.argsort()
        return token_counts_by_expert, gather_indices, scatter_indices
    else:
        return token_counts_by_expert, gather_indices
</syntaxhighlight>

'''Function: torch_grouped_gemm'''

<syntaxhighlight lang="python">
def torch_grouped_gemm(X, W, m_sizes, transpose=True):
    """
    Reference implementation of grouped GEMM.

    X: [M, K] if forward, else [M, N]
    W: [E, N, K]
    m_sizes: [E]

    Returns:
        Y: [M, N] if forward, else [M, K]
    """
    M, K = X.shape
    E = m_sizes.shape[0]
    N = W.shape[1]

    result = torch.zeros((M, N), dtype=X.dtype, device=X.device)

    m_start = 0
    for g in range(E):
        m_size = m_sizes[g]
        if m_size > 0:
            m_end = m_start + m_size
            X_g = X[m_start:m_end]  # [m_size, K]
            W_g = W[g]              # [N, K]
            W_g = W_g.T if transpose else W_g
            Y_g = X_g @ W_g
            result[m_start:m_end] = Y_g
            m_start = m_end
    return result
</syntaxhighlight>

== I/O Contract ==

'''permute:'''
{| class="wikitable"
|-
! Parameter !! Type !! Shape !! Description
|-
| X || torch.Tensor || (num_tokens, hidden_dim) || Input hidden states in token order
|-
| gather_indices || torch.Tensor || (num_tokens * topk,) || Indices for reordering to expert order
|-
| topk || int || scalar || Number of experts per token
|-
| '''Returns''' || torch.Tensor || (num_tokens * topk, hidden_dim) || Hidden states in expert order
|}

'''unpermute:'''
{| class="wikitable"
|-
! Parameter !! Type !! Shape !! Description
|-
| X || torch.Tensor || (num_tokens * topk, hidden_dim) || Hidden states in expert order
|-
| gather_indices || torch.Tensor || (num_tokens * topk,) || Original gather indices
|-
| '''Returns''' || torch.Tensor || (num_tokens * topk, hidden_dim) || Hidden states in token order
|}

'''calculate_topk:'''
{| class="wikitable"
|-
! Parameter !! Type !! Default !! Description
|-
| gating_output || torch.Tensor || required || Router logits (num_tokens, num_experts)
|-
| top_k || int || required || Number of experts to select per token
|-
| use_sigmoid || bool || required || Use sigmoid (True) or softmax (False) activation
|-
| renormalize || bool || required || Normalize top-k weights to sum to 1
|-
| pre_act || bool || True || Apply activation before top-k selection
|-
| post_act || bool || False || Apply activation after top-k selection
|-
| '''Returns''' || Tuple || - || (topk_weights, topk_ids) both shape (num_tokens, top_k)
|}

'''get_routing_indices:'''
{| class="wikitable"
|-
! Parameter !! Type !! Default !! Description
|-
| selected_experts || torch.Tensor || required || Expert IDs per token (num_tokens, top_k) or flattened
|-
| num_experts || int || required || Total number of experts
|-
| return_scatter_indices || bool || False || Also return inverse permutation indices
|-
| '''Returns''' || Tuple || - || (token_counts_by_expert, gather_indices, [scatter_indices])
|}

'''torch_grouped_gemm:'''
{| class="wikitable"
|-
! Parameter !! Type !! Shape !! Description
|-
| X || torch.Tensor || (M, K) || Input activations ordered by expert
|-
| W || torch.Tensor || (E, N, K) || Expert weight matrices
|-
| m_sizes || torch.Tensor || (E,) || Token count per expert
|-
| transpose || bool || True || Whether to transpose W before multiplication
|-
| '''Returns''' || torch.Tensor || (M, N) || Output activations
|}

== Usage Examples ==

'''Token routing workflow:'''

<syntaxhighlight lang="python">
import torch
from grouped_gemm.reference.moe_ops import (
    permute, unpermute, get_routing_indices, torch_grouped_gemm, calculate_topk
)

# Setup
batch_size, seq_len, hidden_dim = 2, 512, 2048
num_experts, top_k = 8, 2
hidden_states = torch.randn(batch_size * seq_len, hidden_dim, device="cuda")

# Simulate router output
router_logits = torch.randn(batch_size * seq_len, num_experts, device="cuda")

# Calculate top-k experts with softmax
topk_weights, topk_ids = calculate_topk(
    router_logits,
    top_k=top_k,
    use_sigmoid=False,  # Use softmax
    renormalize=True,
    pre_act=True
)

# Get routing indices
token_counts, gather_indices = get_routing_indices(topk_ids, num_experts)

# Permute to expert order
permuted_states = permute(hidden_states, gather_indices, top_k)

# After expert computation, unpermute back to token order
output_states = unpermute(permuted_states, gather_indices)
</syntaxhighlight>

'''Grouped GEMM computation:'''

<syntaxhighlight lang="python">
# Expert weights: [num_experts, output_dim, input_dim]
expert_weights = torch.randn(num_experts, 4096, hidden_dim, device="cuda")

# Permuted hidden states already in expert order
permuted_states = permute(hidden_states, gather_indices, top_k)

# Grouped GEMM - processes each expert's tokens as a batch
output = torch_grouped_gemm(
    X=permuted_states,
    W=expert_weights,
    m_sizes=token_counts,
    transpose=True  # W is [N, K], need W.T for X @ W.T
)
</syntaxhighlight>

'''Sigmoid vs Softmax routing:'''

<syntaxhighlight lang="python">
# Llama4-style sigmoid routing (pre_act)
topk_weights, topk_ids = calculate_topk(
    router_logits,
    top_k=2,
    use_sigmoid=True,
    renormalize=False,
    pre_act=True
)

# Qwen3-style softmax routing with normalization
topk_weights, topk_ids = calculate_topk(
    router_logits,
    top_k=2,
    use_sigmoid=False,
    renormalize=True,
    pre_act=True
)

# Triton-bench style post-activation routing
topk_weights, topk_ids = calculate_topk(
    router_logits,
    top_k=2,
    use_sigmoid=True,
    renormalize=False,
    pre_act=False,
    post_act=True
)
</syntaxhighlight>

'''Getting scatter indices for inverse permutation:'''

<syntaxhighlight lang="python">
# Get both gather and scatter indices
token_counts, gather_indices, scatter_indices = get_routing_indices(
    topk_ids,
    num_experts,
    return_scatter_indices=True
)

# scatter_indices can be used for alternative unpermute implementation
# unpermuted[scatter_indices] = permuted is equivalent to unpermute()
</syntaxhighlight>

== Related Pages ==

* [[Implementation:Unslothai_Unsloth_Llama4_MoE_Layer]] - Llama4 MoE layer using these operations
* [[Implementation:Unslothai_Unsloth_Qwen3_MoE_Layer]] - Qwen3 MoE layer using these operations
* [[Implementation:Unslothai_Unsloth_MoE_Block]] - Triton MoE block implementation
