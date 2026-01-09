# Implementation: GEGLU_Kernels

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
Triton kernels for GEGLU (Gated Gaussian Error Linear Unit) activation functions with forward and backward passes.

=== Description ===
This module provides optimized Triton implementations of GEGLU activation used in modern transformer FFN layers. It includes both exact (using erf) and approximate (using tanh) variants, matching the HuggingFace implementations. The kernels handle both forward and backward passes with fused operations for efficiency.

=== Usage ===
Use these kernels when training or running inference on models with GEGLU activation (e.g., LLaMA, Mistral, Phi). They're automatically used by Unsloth's model patches.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/geglu.py unsloth/kernels/geglu.py]
* '''Lines:''' 1-290

=== Key Functions ===
<syntaxhighlight lang="python">
def geglu_exact_forward_kernel(
    gate: torch.Tensor,   # [batch, seq_len, hidden_dim]
    up: torch.Tensor,     # [batch, seq_len, hidden_dim]
) -> torch.Tensor:        # [batch, seq_len, hidden_dim]
    """
    Exact GEGLU forward using erf.
    f = 0.5 * gate * (1 + erf(gate / sqrt(2)))
    output = f * up
    """

def geglu_exact_backward_kernel(
    DW: torch.Tensor,     # Gradient from upstream
    e: torch.Tensor,      # Gate values (modified in-place)
    g: torch.Tensor,      # Up values (modified in-place)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact GEGLU backward pass with fused derivative computation."""

def geglu_approx_forward_kernel(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate GEGLU forward using tanh (faster).
    f = 0.5 * gate * (1 + tanh(sqrt(2/pi) * gate * (1 + 0.044715 * gate^2)))
    output = f * up
    """

def geglu_approx_backward_kernel(
    DW: torch.Tensor,
    e: torch.Tensor,
    g: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate GEGLU backward with tanh derivative."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.geglu import (
    geglu_exact_forward_kernel,
    geglu_exact_backward_kernel,
    geglu_approx_forward_kernel,
    geglu_approx_backward_kernel,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs (Forward) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| gate || Tensor || Yes || Gate projection [batch, seq_len, hidden_dim]
|-
| up || Tensor || Yes || Up projection [batch, seq_len, hidden_dim]
|}

=== Inputs (Backward) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| DW || Tensor || Yes || Upstream gradient (modified in-place with h = f*g)
|-
| e || Tensor || Yes || Gate tensor (modified in-place with df = DW*f)
|-
| g || Tensor || Yes || Up tensor (modified in-place with de)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (forward) || Tensor || GEGLU output [batch, seq_len, hidden_dim]
|-
| (backward) || Tuple || (h, df, de) stored in-place in input buffers
|}

== Mathematical Details ==

=== Exact GEGLU (using erf) ===
<syntaxhighlight lang="text">
Forward:
  f = 0.5 * x * (1 + erf(x / sqrt(2)))
  h = f * up

Backward:
  df/dx = 0.5 * (1 + erf(x/sqrt(2))) + (1/sqrt(2*pi)) * x * exp(-0.5 * x^2)
</syntaxhighlight>

=== Approximate GEGLU (using tanh) ===
<syntaxhighlight lang="text">
Forward:
  s = sqrt(2/pi) = 0.7978845608...
  f = 0.5 * x * (1 + tanh(s * x * (1 + 0.044715 * x^2)))
  h = f * up

Backward:
  Uses sech^2(x) = 1 - tanh^2(x) for efficient derivative computation
</syntaxhighlight>

== Usage Examples ==

=== Forward Pass ===
<syntaxhighlight lang="python">
from unsloth.kernels.geglu import geglu_exact_forward_kernel
import torch

# In an MLP forward pass
batch, seq_len, hidden = 2, 512, 4096

# After linear projections
gate = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
up = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.bfloat16)

# GEGLU activation
output = geglu_exact_forward_kernel(gate, up)
# output shape: [2, 512, 4096]
</syntaxhighlight>

=== Custom Autograd Function ===
<syntaxhighlight lang="python">
from unsloth.kernels.geglu import (
    geglu_exact_forward_kernel,
    geglu_exact_backward_kernel,
)

class GEGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        output = geglu_exact_forward_kernel(gate, up)
        ctx.save_for_backward(gate, up)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        # Clone for in-place modification
        DW = grad_output.clone()
        e = gate.clone()
        g = up.clone()
        geglu_exact_backward_kernel(DW, e, g)
        # e now contains d_gate, g contains d_up
        return e, g
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
