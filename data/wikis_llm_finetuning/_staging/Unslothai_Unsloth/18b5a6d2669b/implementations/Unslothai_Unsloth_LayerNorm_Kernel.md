# Implementation: LayerNorm_Kernel

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Blog|llm.c LayerNorm|https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md]]
|-
! Domains
| [[domain::Kernels]], [[domain::Normalization]], [[domain::Triton]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Triton-based Layer Normalization kernel with optimized forward and backward passes for FP32 precision.

=== Description ===
This module provides an optimized Triton implementation of Layer Normalization that maintains FP32 precision for numerical stability (matching torchtune's Fp32LayerNorm). It includes both forward and backward kernels with efficient memory access patterns and supports the full autograd interface.

=== Usage ===
Use this kernel as a drop-in replacement for `torch.nn.LayerNorm` when you need faster layer normalization with guaranteed FP32 internal computation.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/layernorm.py unsloth/kernels/layernorm.py]
* '''Lines:''' 1-225

=== Key Functions ===
<syntaxhighlight lang="python">
class Fast_Layernorm(torch.autograd.Function):
    """Autograd function for fast LayerNorm with Triton kernels."""

    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,      # Input tensor [*, hidden_dim]
        W: torch.Tensor,      # Weight (gamma) [hidden_dim]
        b: torch.Tensor,      # Bias (beta) [hidden_dim]
        eps: float,           # Epsilon for numerical stability
    ) -> torch.Tensor:
        """Forward pass computing LayerNorm with FP32 precision."""

    @staticmethod
    def backward(ctx, dY: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """Backward pass following Karpathy's llm.c derivation."""


def fast_layernorm(
    layernorm: nn.LayerNorm,  # PyTorch LayerNorm module
    X: torch.Tensor,           # Input tensor
) -> torch.Tensor:
    """
    Apply fast LayerNorm using Triton kernels.

    Args:
        layernorm: nn.LayerNorm instance (must have elementwise_affine=True)
        X: Input tensor of any shape with last dim matching layernorm

    Returns:
        Normalized tensor with same shape as X
    """


def test_layernorm(
    dim: int = 1024,
    eps: float = 1e-5,
    dtype: torch.dtype = torch.float16,
    bsz: int = 21,
    random_state: int = 3407,
    seqlen: int = 3341,
) -> None:
    """Test function verifying gradient correctness."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.layernorm import fast_layernorm, Fast_Layernorm
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| X || Tensor || Yes || Input tensor [..., hidden_dim]
|-
| W || Tensor || Yes || Weight/gamma parameter [hidden_dim]
|-
| b || Tensor || Yes || Bias/beta parameter [hidden_dim]
|-
| eps || float || Yes || Epsilon (typically 1e-5 or 1e-6)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output || Tensor || Normalized tensor with same shape as X
|}

== Mathematical Details ==

<syntaxhighlight lang="text">
Forward:
  mean = sum(X) / n_cols
  var = sum((X - mean)^2) / n_cols
  inv_var = rsqrt(var + eps)
  Y = (X - mean) * inv_var * W + b

Backward (following llm.c):
  normed = (X - mean) * inv_var
  dY_W = dY * W
  dX = inv_var * (dY_W - mean(dY_W) - normed * mean(dY_W * normed))
</syntaxhighlight>

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from unsloth.kernels.layernorm import fast_layernorm
import torch.nn as nn

# Create standard LayerNorm
layernorm = nn.LayerNorm(4096, eps=1e-5, device="cuda", dtype=torch.float16)

# Input tensor
x = torch.randn(2, 512, 4096, device="cuda", dtype=torch.float16)

# Apply fast LayerNorm
output = fast_layernorm(layernorm, x)
# Equivalent to layernorm(x) but faster with FP32 internals
</syntaxhighlight>

=== Direct Autograd Usage ===
<syntaxhighlight lang="python">
from unsloth.kernels.layernorm import Fast_Layernorm

# Get layernorm parameters
weight = layernorm.weight
bias = layernorm.bias
eps = layernorm.eps

# Direct call
output = Fast_Layernorm.apply(x, weight, bias, eps)
</syntaxhighlight>

=== Verify Correctness ===
<syntaxhighlight lang="python">
from unsloth.kernels.layernorm import test_layernorm, testing_suite_layernorm

# Single test
test_layernorm(dim=1024, dtype=torch.bfloat16)

# Full test suite
testing_suite_layernorm()  # Tests multiple dims, dtypes, seqlens
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
