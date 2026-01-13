# Implementation: Llama4 MoE Layer

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

The Llama4 MoE Layer module provides reference implementations of the Llama4 Mixture-of-Experts (MoE) text block using grouped GEMM operations. This file contains two primary classes that implement the HuggingFace `Llama4TextMoe` block with optimized grouped matrix multiplication:

* '''Llama4GroupedGemmTextMoe''': A torch-native grouped GEMM implementation that serves as a reference baseline
* '''Llama4TritonTextMoe''': A high-performance implementation using Triton grouped GEMM kernels

The module supports optional overlapping of router and shared expert computations using CUDA streams for improved performance. It also includes a dataclass `Llama4MoeResult` for capturing intermediate computation results during debugging.

== Code Reference ==

'''File Path:''' `unsloth/kernels/moe/grouped_gemm/reference/layers/llama4_moe.py`

'''Key Classes:'''

<syntaxhighlight lang="python">
@dataclass
class Llama4MoeResult:
    token_counts_by_expert: torch.Tensor
    gather_indices: torch.Tensor
    topk_weights: torch.Tensor
    hidden_states_after_weight_merge: torch.Tensor
    first_gemm: torch.Tensor
    intermediate: torch.Tensor
    second_gemm: torch.Tensor
    hidden_states_unpermute: torch.Tensor
    shared_expert_out: torch.Tensor
    final_out: torch.Tensor
    router_logits: torch.Tensor = None
</syntaxhighlight>

'''Llama4GroupedGemmTextMoe''' - Extends `Llama4TextMoe` with torch-native grouped GEMM:

<syntaxhighlight lang="python">
class Llama4GroupedGemmTextMoe(Llama4TextMoe):
    EXPERT_WEIGHT_NAMES = ["experts.gate_up_proj", "experts.down_proj"]

    def __init__(
        self,
        config: Llama4TextConfig,
        overlap_router_shared=False,
        verbose=False,
        debug=False,
    ):
        # Permutes expert weights in-place for optimal memory layout
        # Sets up CUDA streams for overlapped computation when enabled
</syntaxhighlight>

'''Llama4TritonTextMoe''' - Extends the grouped GEMM implementation with Triton kernels:

<syntaxhighlight lang="python">
class Llama4TritonTextMoe(Llama4GroupedGemmTextMoe):
    def __init__(
        self,
        config: Llama4TextConfig,
        overlap_router_shared=False,
        permute_x: bool = False,
        permute_y: bool = True,
        autotune: bool = True,
        kernel_config_fwd: KernelConfigForward = None,
        kernel_config_bwd_dW: KernelConfigBackward_dW = None,
        kernel_config_bwd_dX: KernelConfigBackward_dX = None,
        dW_only: bool = False,
        dX_only: bool = False,
        verbose=False,
    ):
</syntaxhighlight>

'''Key Methods:'''
* `copy_weights(other)`: Copies weights from a standard Llama4TextMoe block with proper permutation
* `check_weights(other)`: Validates weight consistency between implementations
* `act_and_mul(x)`: Applies SiLU activation and element-wise multiplication (gate * up projection)
* `run_router(hidden_states)`: Computes router logits and selects top-k experts using sigmoid activation
* `get_token_counts_and_gather_indices(selected_experts)`: Computes routing indices for token permutation
* `forward(hidden_states)`: Main forward pass implementing the full MoE computation

== I/O Contract ==

'''Input:'''
{| class="wikitable"
|-
! Parameter !! Type !! Shape !! Description
|-
| hidden_states || torch.Tensor || (batch_size, sequence_length, hidden_dim) || Input hidden states from previous layer
|}

'''Output:'''
{| class="wikitable"
|-
! Output !! Type !! Shape !! Description
|-
| final_out / hidden_states || torch.Tensor || (batch_size, sequence_length, hidden_dim) || MoE processed output combined with shared expert
|-
| routing_weights || torch.Tensor || (num_tokens, top_k) || Top-k routing weights after sigmoid activation
|}

'''Debug Mode Output (Llama4MoeResult):'''
{| class="wikitable"
|-
! Field !! Shape !! Description
|-
| token_counts_by_expert || (num_experts,) || Token count per expert
|-
| gather_indices || (num_tokens * top_k,) || Indices for token permutation
|-
| topk_weights || (num_tokens, top_k) || Routing weights
|-
| hidden_states_after_weight_merge || (num_tokens, hidden_dim) || States after weight merging
|-
| first_gemm || (total_tokens, 2 * expert_dim) || Output of first grouped GEMM (gate_up_proj)
|-
| intermediate || (total_tokens, expert_dim) || Output after activation function
|-
| second_gemm || (total_tokens, hidden_dim) || Output of second grouped GEMM (down_proj)
|-
| hidden_states_unpermute || (total_tokens, hidden_dim) || States after unpermutation
|-
| shared_expert_out || (num_tokens, hidden_dim) || Output from shared expert
|-
| final_out || (num_tokens, hidden_dim) || Combined final output
|}

'''Configuration Parameters:'''
* `overlap_router_shared`: Enable CUDA stream overlap for router and shared expert computation
* `permute_x` (Triton only): Fuse input permutation in kernel prologue (disabled for Llama4 due to pre-multiplication)
* `permute_y` (Triton only): Fuse output unpermutation in kernel epilogue
* `autotune`: Enable Triton kernel autotuning
* `dW_only` / `dX_only`: Control gradient computation for weight or input gradients only

== Usage Examples ==

'''Creating from HuggingFace Llama4TextMoe:'''

<syntaxhighlight lang="python">
from transformers.models.llama4 import Llama4TextConfig
from grouped_gemm.reference.layers.llama4_moe import (
    Llama4GroupedGemmTextMoe,
    Llama4TritonTextMoe
)

# Load config and create standard HF MoE block
config = Llama4TextConfig(...)
hf_moe = Llama4TextMoe(config)

# Create torch-native grouped GEMM version
grouped_moe = Llama4GroupedGemmTextMoe(config, overlap_router_shared=True)
grouped_moe.copy_weights(hf_moe)

# Create Triton-accelerated version
triton_moe = Llama4TritonTextMoe(
    config,
    overlap_router_shared=True,
    permute_y=True,
    autotune=True
)
triton_moe.copy_weights(hf_moe)
</syntaxhighlight>

'''Forward pass:'''

<syntaxhighlight lang="python">
# Input: hidden_states with shape [batch_size, seq_len, hidden_dim]
hidden_states = torch.randn(2, 512, 4096, device="cuda", dtype=torch.bfloat16)

# Standard forward
output, routing_weights = triton_moe(hidden_states)

# Debug mode forward (grouped GEMM version only)
grouped_moe.debug = True
result = grouped_moe(hidden_states)
# Access intermediate results: result.first_gemm, result.intermediate, etc.
</syntaxhighlight>

'''Weight validation:'''

<syntaxhighlight lang="python">
# Verify weights were copied correctly
triton_moe.check_weights(hf_moe)
</syntaxhighlight>

== Related Pages ==

* [[Implementation:Unslothai_Unsloth_MoE_Ops]] - Common MoE operations used by this layer
* [[Implementation:Unslothai_Unsloth_MoE_Block]] - Generic Triton MoE block implementation
* [[Implementation:Unslothai_Unsloth_Qwen3_MoE_Layer]] - Similar MoE layer for Qwen3 architecture
