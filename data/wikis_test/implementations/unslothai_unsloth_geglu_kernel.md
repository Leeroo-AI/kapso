# Implementation: GEGLU Kernel

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|GLU Variants Improve Transformer|https://arxiv.org/abs/2002.05202]]
* [[source::Doc|Triton Documentation|https://triton-lang.org/main/index.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Kernels]], [[domain::Activation_Functions]], [[domain::Triton]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
Optimized Triton kernels implementing GEGLU (Gaussian Error Gated Linear Unit) activation with both exact and approximate computation modes, including forward and backward passes.

=== Description ===
The GEGLU (Gated Error Linear Unit with Gaussian activation) kernel provides high-performance implementations of the GEGLU activation function used in transformer MLP layers. This activation combines gating mechanisms with GELU nonlinearity for improved model capacity.

The implementation provides two variants:
1. **Exact GEGLU** - Uses `erf()` function for precise GELU computation: `f = 0.5 * x * (1 + erf(x/√2))`
2. **Approximate GEGLU** - Uses `tanh()` approximation for faster computation: `f = 0.5 * x * (1 + tanh(√(2/π) * x * (1 + 0.044715 * x²)))`

Both variants implement fused forward and backward kernels that:
- Handle element-wise multiplication with the gate projection
- Support large tensors (>2³¹ elements) via int64 indexing
- Maintain numerical stability through careful dtype handling

GEGLU is notably used in Google's Gemma models and other transformer variants that require improved gradient flow compared to standard ReLU/GELU activations.

=== Usage ===
Import this kernel when working with models that use GEGLU activation (e.g., Gemma, GLM variants). The kernel is automatically invoked by Unsloth's model patching system when loading compatible models.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/geglu.py#L63-L78 unsloth/kernels/geglu.py]
* '''Lines:''' 63-78 (exact forward), 140-153 (exact backward), 193-208 (approx forward), 277-290 (approx backward)

=== Signature ===
<syntaxhighlight lang="python">
def geglu_exact_forward_kernel(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Exact GEGLU forward pass using erf function.

    Args:
        gate: Gate projection tensor, shape (batch, seq_len, hidden_dim)
        up: Up projection tensor, shape (batch, seq_len, hidden_dim)

    Returns:
        Output tensor with GEGLU activation applied, same shape as inputs
    """

def geglu_exact_backward_kernel(DW: torch.Tensor, e: torch.Tensor, g: torch.Tensor) -> tuple:
    """
    Exact GEGLU backward pass computing gradients in-place.

    Args:
        DW: Upstream gradient tensor
        e: Gate activation values (modified in-place)
        g: Up projection values (modified in-place)

    Returns:
        Tuple of (h, df, de) written in-place to input buffers
    """

def geglu_approx_forward_kernel(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Approximate GEGLU forward using tanh approximation."""

def geglu_approx_backward_kernel(DW: torch.Tensor, e: torch.Tensor, g: torch.Tensor) -> tuple:
    """Approximate GEGLU backward pass."""
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

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| gate || torch.Tensor || Yes || Gate projection output from first linear layer, shape (batch, seq_len, hidden_dim)
|-
| up || torch.Tensor || Yes || Up projection output from second linear layer, same shape as gate
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output || torch.Tensor || GEGLU activation result: `gelu(gate) * up`, same shape as inputs
|}

== Usage Examples ==

=== Direct Kernel Usage ===
<syntaxhighlight lang="python">
import torch
from unsloth.kernels.geglu import geglu_exact_forward_kernel, geglu_approx_forward_kernel

# Simulating MLP projections
batch, seq_len, hidden = 2, 512, 4096
gate = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
up = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.bfloat16)

# Use exact GEGLU (more accurate)
output_exact = geglu_exact_forward_kernel(gate, up)

# Use approximate GEGLU (faster)
output_approx = geglu_approx_forward_kernel(gate, up)

# Verify shapes
assert output_exact.shape == (batch, seq_len, hidden)
</syntaxhighlight>

=== Integration with Gemma Model ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load Gemma model - GEGLU kernels are automatically used
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# The GEGLU kernels are invoked internally during forward/backward passes
# No manual intervention needed - Unsloth patches the model automatically
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:unslothai_unsloth_GPU_CUDA_Environment]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_Dtype_Selection]]
