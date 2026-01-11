# Implementation: SwiGLU_Kernels

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|GLU Variants|https://arxiv.org/abs/2002.05202]]
|-
! Domains
| [[domain::Kernels]], [[domain::Activation]], [[domain::Triton]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Triton kernels for SwiGLU (Swish-Gated Linear Unit) activation functions used in LLaMA-style transformer FFN layers.

=== Description ===
This module provides optimized Triton implementations of SwiGLU activation, which combines the Swish activation (x * sigmoid(x)) with gating. It includes both forward and backward passes with fused operations. SwiGLU is used in LLaMA, Mistral, and other modern LLMs.

=== Usage ===
Use these kernels when training or running inference on models with SwiGLU activation in their MLP layers. They're automatically used by Unsloth's model patches.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/swiglu.py unsloth/kernels/swiglu.py]
* '''Lines:''' 1-143

=== Key Functions ===
<syntaxhighlight lang="python">
def swiglu_fg_kernel(
    e: torch.Tensor,   # Gate projection [batch, seq_len, hidden_dim]
    g: torch.Tensor,   # Up projection [batch, seq_len, hidden_dim]
) -> torch.Tensor:     # Output [batch, seq_len, hidden_dim]
    """
    SwiGLU forward pass.
    f = e * sigmoid(e)  # Swish activation
    h = f * g           # Gating
    """

def swiglu_DWf_DW_dfg_kernel(
    DW: torch.Tensor,  # Upstream gradient (modified in-place with h)
    e: torch.Tensor,   # Gate tensor (modified in-place with df)
    g: torch.Tensor,   # Up tensor (modified in-place with de)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SwiGLU backward pass with fused derivative computation.
    Stores results in-place in input buffers.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.swiglu import (
    swiglu_fg_kernel,
    swiglu_DWf_DW_dfg_kernel,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs (Forward) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| e || Tensor || Yes || Gate projection [batch, seq_len, hidden_dim]
|-
| g || Tensor || Yes || Up projection [batch, seq_len, hidden_dim]
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| h || Tensor || SwiGLU output [batch, seq_len, hidden_dim]
|}

== Mathematical Details ==

<syntaxhighlight lang="text">
Forward:
  se = sigmoid(e) = 1 / (1 + exp(-e))
  f = e * se       # Swish activation
  h = f * g        # Gating

Backward:
  df/de = se * (1 + e * (1 - se))
  d_gate = dh * g * df/de
  d_up = dh * f
</syntaxhighlight>

== Usage Examples ==

=== Forward Pass ===
<syntaxhighlight lang="python">
from unsloth.kernels.swiglu import swiglu_fg_kernel
import torch

# In an MLP forward pass
batch, seq_len, hidden = 2, 512, 4096

# After linear projections
gate = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
up = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.bfloat16)

# SwiGLU activation
output = swiglu_fg_kernel(gate, up)
# output shape: [2, 512, 4096]
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
