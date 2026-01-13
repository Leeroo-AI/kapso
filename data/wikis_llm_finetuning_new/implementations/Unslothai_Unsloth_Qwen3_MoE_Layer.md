# Implementation: Qwen3 MoE Layer

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

The Qwen3 MoE Layer module provides reference implementations of the Qwen3 Mixture-of-Experts sparse block using grouped GEMM operations. This module contains two primary classes:

* '''Qwen3MoeGroupedGEMMBlock''': A torch-native grouped GEMM reference implementation for debugging and validation
* '''Qwen3MoeFusedGroupedGEMMBlock''': A high-performance implementation using Triton grouped GEMM kernels with optional fused permutation operations

Unlike Llama4's MoE implementation, Qwen3 uses softmax-based routing with optional top-k probability normalization. The module includes utilities for extracting and converting weights from HuggingFace Qwen3MoeSparseMoeBlock format.

== Code Reference ==

'''File Path:''' `unsloth/kernels/moe/grouped_gemm/reference/layers/qwen3_moe.py`

'''Key Classes:'''

<syntaxhighlight lang="python">
@dataclass
class GroupedGEMMResult:
    token_counts_by_expert: torch.Tensor
    gather_indices: torch.Tensor
    topk_weights: torch.Tensor
    first_gemm: torch.Tensor
    intermediate: torch.Tensor
    second_gemm: torch.Tensor
    hidden_states_unpermute: torch.Tensor
    hidden_states: torch.Tensor  # final output
</syntaxhighlight>

'''Qwen3MoeGroupedGEMMBlock''' - Torch-native reference implementation:

<syntaxhighlight lang="python">
class Qwen3MoeGroupedGEMMBlock(torch.nn.Module):
    def __init__(
        self,
        config,
        gate: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ):
        # gate: [num_experts, hidden_size] - router weights
        # gate_up_proj: [num_experts, 2 * moe_intermediate_size, hidden_size]
        # down_proj: [num_experts, hidden_size, moe_intermediate_size]
</syntaxhighlight>

'''Qwen3MoeFusedGroupedGEMMBlock''' - Triton-accelerated implementation:

<syntaxhighlight lang="python">
class Qwen3MoeFusedGroupedGEMMBlock(Qwen3MoeGroupedGEMMBlock):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        gate: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        permute_x: bool = True,
        permute_y: bool = True,
        autotune: bool = True,
        kernel_config_fwd: KernelConfigForward = None,
        kernel_config_bwd_dW: KernelConfigBackward_dW = None,
        kernel_config_bwd_dX: KernelConfigBackward_dX = None,
        dW_only: bool = False,
        dX_only: bool = False,
    ):
</syntaxhighlight>

'''Key Methods:'''
* `extract_hf_weights(moe_block)`: Static method to extract and convert weights from HuggingFace format
* `from_hf(moe_block)`: Class method to create instance from HuggingFace Qwen3MoeSparseMoeBlock
* `check_weights(moe_block)`: Validates weight consistency with HuggingFace block
* `act_and_mul(x)`: Applies activation function and gate-up multiplication
* `run_router(hidden_states)`: Computes softmax routing with optional normalization
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

'''Output (Reference Implementation):'''
{| class="wikitable"
|-
! Output !! Type !! Description
|-
| GroupedGEMMResult || dataclass || Contains all intermediate results for debugging
|-
| router_logits || torch.Tensor || Raw router logits before softmax
|}

'''Output (Fused Implementation):'''
{| class="wikitable"
|-
! Output !! Type !! Shape !! Description
|-
| hidden_states || torch.Tensor || (batch_size, sequence_length, hidden_dim) || MoE processed output
|-
| router_logits || torch.Tensor || (num_tokens, num_experts) || Raw router logits
|}

'''GroupedGEMMResult Fields:'''
{| class="wikitable"
|-
! Field !! Shape !! Description
|-
| token_counts_by_expert || (num_experts,) || Token count assigned to each expert
|-
| gather_indices || (num_tokens * top_k,) || Indices for token-to-expert permutation
|-
| topk_weights || (num_tokens, top_k) || Normalized routing weights for selected experts
|-
| first_gemm || (total_tokens, 2 * moe_intermediate_size) || Output of gate_up_proj GEMM
|-
| intermediate || (total_tokens, moe_intermediate_size) || Output after activation (gate * up)
|-
| second_gemm || (total_tokens, hidden_dim) || Output of down_proj GEMM
|-
| hidden_states_unpermute || (total_tokens, hidden_dim) || States after unpermutation
|-
| hidden_states || (batch_size, sequence_length, hidden_dim) || Final weighted output
|}

'''Configuration Parameters:'''
* `norm_topk_prob`: Whether to normalize top-k probabilities (Qwen3-specific feature)
* `permute_x`: Fuse input permutation in kernel prologue
* `permute_y`: Fuse output unpermutation in kernel epilogue
* `autotune`: Enable Triton kernel autotuning
* `dW_only` / `dX_only`: Control gradient computation scope

== Usage Examples ==

'''Creating from HuggingFace Qwen3MoeSparseMoeBlock:'''

<syntaxhighlight lang="python">
from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from grouped_gemm.reference.layers.qwen3_moe import (
    Qwen3MoeGroupedGEMMBlock,
    Qwen3MoeFusedGroupedGEMMBlock
)

# Load HuggingFace MoE block
config = Qwen3MoeConfig(...)
hf_moe = Qwen3MoeSparseMoeBlock(config)

# Create torch-native reference implementation
reference_moe = Qwen3MoeGroupedGEMMBlock.from_hf(hf_moe)

# Create Triton-accelerated implementation
fused_moe = Qwen3MoeFusedGroupedGEMMBlock.from_hf(
    hf_moe,
    permute_x=True,
    permute_y=True,
    autotune=True
)
</syntaxhighlight>

'''Manual construction with extracted weights:'''

<syntaxhighlight lang="python">
# Extract weights from HuggingFace format
gate, gate_up_proj, down_proj = Qwen3MoeGroupedGEMMBlock.extract_hf_weights(hf_moe)

# Create block manually
moe_block = Qwen3MoeGroupedGEMMBlock(
    config,
    gate=gate,
    gate_up_proj=gate_up_proj,
    down_proj=down_proj
)
</syntaxhighlight>

'''Forward pass:'''

<syntaxhighlight lang="python">
# Input: hidden_states with shape [batch_size, seq_len, hidden_dim]
hidden_states = torch.randn(2, 512, 2048, device="cuda", dtype=torch.bfloat16)

# Reference implementation returns full debug info
result, router_logits = reference_moe(hidden_states)
print(f"Token distribution: {result.token_counts_by_expert}")
print(f"First GEMM output shape: {result.first_gemm.shape}")
print(f"Final output shape: {result.hidden_states.shape}")

# Fused implementation returns output and logits
output, router_logits = fused_moe(hidden_states)
</syntaxhighlight>

'''Weight validation:'''

<syntaxhighlight lang="python">
# Verify extracted weights match original HF block
reference_moe.check_weights(hf_moe)
</syntaxhighlight>

== Related Pages ==

* [[Implementation:Unslothai_Unsloth_MoE_Ops]] - Common MoE operations (routing, permutation)
* [[Implementation:Unslothai_Unsloth_MoE_Block]] - Generic Triton MoE block implementation
* [[Implementation:Unslothai_Unsloth_Llama4_MoE_Layer]] - Similar MoE layer for Llama4 architecture
