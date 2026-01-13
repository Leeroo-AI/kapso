# Implementation: MoE Block

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

The MoE Block module provides a Triton-accelerated implementation of the Qwen3 MoE sparse block using grouped GEMM operations. This module extends the `Qwen3MoeGroupedGEMMBlock` base class and replaces the torch-native grouped GEMM with high-performance Triton kernels.

The implementation supports:
* Fused input permutation (`permute_x`) in the kernel prologue
* Fused output unpermutation (`permute_y`) in the kernel epilogue
* Configurable kernel autotuning or manual kernel configuration
* Selective gradient computation (`dW_only`, `dX_only`) for memory optimization

'''Note:''' This is a reference implementation intended for debugging and validation, not production use. It contains additional checks and saves intermediate results.

== Code Reference ==

'''File Path:''' `unsloth/kernels/moe/grouped_gemm/reference/moe_block.py`

'''Main Class:'''

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

* `from_hf(moe_block, ...)`: Class method to create instance from HuggingFace Qwen3MoeSparseMoeBlock with configurable Triton options
* `forward(hidden_states)`: Main forward pass with Triton grouped GEMM kernels

'''Forward Pass Implementation:'''

<syntaxhighlight lang="python">
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    num_tokens = batch_size * sequence_length
    total_tokens = num_tokens * self.top_k

    hidden_states = hidden_states.view(-1, hidden_dim)

    # Router computation
    router_logits, routing_weights, selected_experts = self.run_router(hidden_states)

    # Get routing indices
    token_counts_by_expert, gather_indices = (
        self.get_token_counts_and_gather_indices(selected_experts)
    )

    # Optional input permutation (if not fused)
    if not self.permute_x:
        hidden_states = permute(hidden_states, gather_indices, self.top_k)

    # First grouped GEMM: gate_up_proj
    hidden_states = grouped_gemm(
        X=hidden_states,
        W=self.gate_up_proj,
        m_sizes=token_counts_by_expert,
        gather_indices=gather_indices,
        topk=self.top_k,
        permute_x=self.permute_x,
        permute_y=False,  # Never permute output of first GEMM
        ...
    )

    # Activation: SiLU(gate) * up
    hidden_states = self.act_and_mul(hidden_states)

    # Second grouped GEMM: down_proj
    hidden_states = grouped_gemm(
        X=hidden_states,
        W=self.down_proj,
        m_sizes=token_counts_by_expert,
        ...
        permute_y=self.permute_y,
    )

    # Optional output unpermutation (if not fused)
    if not self.permute_y:
        hidden_states = unpermute(hidden_states, gather_indices)

    # Merge top-k weights
    hidden_states = (
        hidden_states.view(num_tokens, self.top_k, hidden_dim)
        * routing_weights[..., None]
    )
    hidden_states = hidden_states.sum(dim=1)

    return hidden_states.view(batch_size, sequence_length, hidden_dim), router_logits
</syntaxhighlight>

== I/O Contract ==

'''Input:'''
{| class="wikitable"
|-
! Parameter !! Type !! Shape !! Description
|-
| hidden_states || torch.Tensor || (batch_size, sequence_length, hidden_dim) || Input hidden states
|}

'''Output:'''
{| class="wikitable"
|-
! Output !! Type !! Shape !! Description
|-
| hidden_states || torch.Tensor || (batch_size, sequence_length, hidden_dim) || MoE processed output
|-
| router_logits || torch.Tensor || (num_tokens, num_experts) || Raw router logits for auxiliary loss computation
|}

'''Constructor Parameters:'''
{| class="wikitable"
|-
! Parameter !! Type !! Default !! Description
|-
| config || Qwen3MoeConfig || required || Model configuration with MoE settings
|-
| gate || torch.Tensor || required || Router weights [num_experts, hidden_size]
|-
| gate_up_proj || torch.Tensor || required || Fused gate and up projection [num_experts, 2*intermediate, hidden_size]
|-
| down_proj || torch.Tensor || required || Down projection [num_experts, hidden_size, intermediate]
|-
| permute_x || bool || True || Fuse input permutation in first GEMM kernel
|-
| permute_y || bool || True || Fuse output unpermutation in second GEMM kernel
|-
| autotune || bool || True || Enable Triton kernel autotuning
|-
| kernel_config_fwd || KernelConfigForward || None || Manual forward kernel configuration
|-
| kernel_config_bwd_dW || KernelConfigBackward_dW || None || Manual backward (dW) kernel configuration
|-
| kernel_config_bwd_dX || KernelConfigBackward_dX || None || Manual backward (dX) kernel configuration
|-
| dW_only || bool || False || Only compute weight gradients in backward pass
|-
| dX_only || bool || False || Only compute input gradients in backward pass
|}

== Usage Examples ==

'''Creating from HuggingFace model:'''

<syntaxhighlight lang="python">
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from grouped_gemm.reference.moe_block import Qwen3MoeFusedGroupedGEMMBlock

# Load HuggingFace MoE block
hf_moe = Qwen3MoeSparseMoeBlock(config)

# Create Triton-accelerated block with full fusion
fused_moe = Qwen3MoeFusedGroupedGEMMBlock.from_hf(
    hf_moe,
    permute_x=True,   # Fuse input permutation
    permute_y=True,   # Fuse output unpermutation
    autotune=True
)
</syntaxhighlight>

'''Forward pass:'''

<syntaxhighlight lang="python">
hidden_states = torch.randn(2, 512, 2048, device="cuda", dtype=torch.bfloat16)

# Forward pass
output, router_logits = fused_moe(hidden_states)

# Use router_logits for load balancing loss if needed
load_balance_loss = compute_load_balance_loss(router_logits)
</syntaxhighlight>

'''Manual kernel configuration (disabling autotune):'''

<syntaxhighlight lang="python">
from grouped_gemm.kernels.tuning import (
    KernelConfigForward,
    KernelConfigBackward_dW,
    KernelConfigBackward_dX
)

# Define custom kernel configurations
fwd_config = KernelConfigForward(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, ...)
bwd_dW_config = KernelConfigBackward_dW(...)
bwd_dX_config = KernelConfigBackward_dX(...)

# Create block with manual configs
fused_moe = Qwen3MoeFusedGroupedGEMMBlock.from_hf(
    hf_moe,
    autotune=False,
    kernel_config_fwd=fwd_config,
    kernel_config_bwd_dW=bwd_dW_config,
    kernel_config_bwd_dX=bwd_dX_config
)
</syntaxhighlight>

'''Memory-efficient gradient computation:'''

<syntaxhighlight lang="python">
# Only compute weight gradients (for fine-tuning frozen activations)
fused_moe = Qwen3MoeFusedGroupedGEMMBlock.from_hf(hf_moe, dW_only=True)

# Only compute input gradients (for frozen weights)
fused_moe = Qwen3MoeFusedGroupedGEMMBlock.from_hf(hf_moe, dX_only=True)
</syntaxhighlight>

== Related Pages ==

* [[Implementation:Unslothai_Unsloth_MoE_Ops]] - Common MoE operations (permute, unpermute, routing)
* [[Implementation:Unslothai_Unsloth_Qwen3_MoE_Layer]] - Full Qwen3 MoE layer with both reference and Triton implementations
* [[Implementation:Unslothai_Unsloth_Llama4_MoE_Layer]] - Llama4 MoE layer implementation
