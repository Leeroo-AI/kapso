# Implementation: GEGLU_Kernels

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Kernels]], [[domain::GPU_Optimization]], [[domain::Activation_Functions]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==
Provides Triton-optimized kernels for GEGLU (Gated Gaussian Error Linear Unit) activation functions, supporting both exact and approximate variants with custom forward and backward passes.

=== Description ===
The GEGLU Kernels module implements high-performance Triton kernels for the GEGLU activation function used in modern transformer architectures like LLaMA and Mistral. The module provides two variants:

# '''Exact GEGLU''': Uses the precise GELU formula with error function: <code>f = 0.5 * x * (1 + erf(x / sqrt(2)))</code>
# '''Approximate GEGLU''': Uses the tanh-based approximation: <code>f = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))</code>

Key optimizations include:
* Fused gate and up projection multiplication in a single kernel
* In-place backward pass to minimize memory allocations
* Automatic long indexing for sequences exceeding INT32 limits (>2B elements)
* Configurable block size (1024 elements per block)
* Float32 intermediate computation for numerical stability

The backward kernels compute gradients for both the gate and up projections simultaneously, reusing intermediate computations to minimize memory bandwidth.

=== Usage ===
This kernel is used in:
* LLaMA-style MLP layers with SwiGLU/GEGLU activation
* Mistral and Mixtral model architectures
* Any transformer using gated linear units with GELU activation
* Training and inference when Unsloth patches are applied

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' unsloth/kernels/geglu.py
* '''Lines:''' 1-290

=== Signature ===
<syntaxhighlight lang="python">
def geglu_exact_forward_kernel(gate, up):
    """
    Compute exact GEGLU forward pass: h = GELU(gate) * up

    Args:
        gate: Gate projection tensor of shape (batch, seq_len, hidden_dim)
        up: Up projection tensor of shape (batch, seq_len, hidden_dim)

    Returns:
        Output tensor of shape (batch, seq_len, hidden_dim)
    """

def geglu_exact_backward_kernel(DW, e, g):
    """
    Compute exact GEGLU backward pass gradients in-place.

    Args:
        DW: Upstream gradient tensor (modified in-place to h = f * g)
        e: Gate tensor (modified in-place to df = DW * f)
        g: Up tensor (modified in-place to de = gradient w.r.t. gate)

    Returns:
        Tuple of (h, df, de) - all computed in-place in input buffers
    """

def geglu_approx_forward_kernel(gate, up):
    """
    Compute approximate GEGLU forward pass using tanh approximation.

    Args:
        gate: Gate projection tensor of shape (batch, seq_len, hidden_dim)
        up: Up projection tensor of shape (batch, seq_len, hidden_dim)

    Returns:
        Output tensor of shape (batch, seq_len, hidden_dim)
    """

def geglu_approx_backward_kernel(DW, e, g):
    """
    Compute approximate GEGLU backward pass gradients in-place.

    Args:
        DW: Upstream gradient tensor
        e: Gate tensor
        g: Up tensor

    Returns:
        Tuple of modified tensors with gradients
    """
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

=== geglu_exact_forward_kernel / geglu_approx_forward_kernel ===

{| class="wikitable"
|+ Input Parameters
|-
! Name !! Type !! Shape !! Description
|-
| gate || torch.Tensor || (batch, seq_len, hidden_dim) || Gate projection from linear layer
|-
| up || torch.Tensor || (batch, seq_len, hidden_dim) || Up projection from linear layer
|}

{| class="wikitable"
|+ Output
|-
! Name !! Type !! Shape !! Description
|-
| output || torch.Tensor || (batch, seq_len, hidden_dim) || GEGLU activation result: GELU(gate) * up
|}

=== geglu_exact_backward_kernel / geglu_approx_backward_kernel ===

{| class="wikitable"
|+ Input Parameters (Modified In-Place)
|-
! Name !! Type !! Shape !! Description
|-
| DW || torch.Tensor || (batch*seq_len, hidden_dim) || Upstream gradient (output: h = f * g)
|-
| e || torch.Tensor || (batch*seq_len, hidden_dim) || Gate tensor (output: df = DW * f)
|-
| g || torch.Tensor || (batch*seq_len, hidden_dim) || Up tensor (output: de = dg * df/de)
|}

{| class="wikitable"
|+ Output (In-Place Modified Inputs)
|-
! Name !! Contains !! Description
|-
| DW || h || Forward output h = f * g
|-
| e || df || Gradient for up projection: DW * f
|-
| g || de || Gradient for gate projection
|}

== Usage Examples ==

=== Exact GEGLU Forward Pass ===
<syntaxhighlight lang="python">
import torch
from unsloth.kernels.geglu import geglu_exact_forward_kernel

# Create gate and up projections
batch, seq_len, hidden_dim = 2, 128, 4096

gate = torch.randn(batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
up = torch.randn(batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)

# Compute GEGLU: output = GELU(gate) * up
output = geglu_exact_forward_kernel(gate, up)
print(f"Output shape: {output.shape}")  # (2, 128, 4096)
</syntaxhighlight>

=== Approximate GEGLU for Faster Computation ===
<syntaxhighlight lang="python">
from unsloth.kernels.geglu import geglu_approx_forward_kernel

# Approximate version is slightly faster with minimal accuracy loss
output = geglu_approx_forward_kernel(gate, up)
</syntaxhighlight>

=== Full Forward-Backward Pass ===
<syntaxhighlight lang="python">
import torch
from unsloth.kernels.geglu import (
    geglu_exact_forward_kernel,
    geglu_exact_backward_kernel,
)

# Forward pass
batch, seq_len, hidden_dim = 2, 128, 4096
gate = torch.randn(batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
up = torch.randn(batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)

output = geglu_exact_forward_kernel(gate, up)

# Backward pass - prepare tensors
# Note: backward operates on flattened (batch*seq_len, hidden_dim) tensors
DW = torch.randn(batch * seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)  # upstream grad
e = gate.view(-1, hidden_dim).clone()  # gate (will be modified)
g = up.view(-1, hidden_dim).clone()    # up (will be modified)

# Compute gradients in-place
h, df, de = geglu_exact_backward_kernel(DW, e, g)
# DW now contains h (forward output)
# e now contains df (gradient for up projection)
# g now contains de (gradient for gate projection)
</syntaxhighlight>

=== Integration with MLP Layer ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from unsloth.kernels.geglu import geglu_exact_forward_kernel

class FastGEGLUMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Use fast Triton kernel for GEGLU
        hidden = geglu_exact_forward_kernel(gate, up)
        return self.down_proj(hidden)
</syntaxhighlight>

== Implementation Details ==

=== Exact GELU Formula ===
The exact GELU uses the error function:

<syntaxhighlight lang="python">
# f = 0.5 * x * (1 + erf(x / sqrt(2)))
f_row = 0.5 * e_row * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
h_row = f_row * g_row  # GEGLU output
</syntaxhighlight>

=== Approximate GELU Formula ===
The approximate version uses tanh for faster computation:

<syntaxhighlight lang="python">
# f = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
s = 0.7978845608028654  # sqrt(2/pi)
f_row = 0.5 * e_row * (triton_tanh(s * e_row * (1.0 + 0.044715 * e_row * e_row)) + 1.0)
</syntaxhighlight>

=== Backward Pass Derivation ===
For the exact backward pass, the gradient is:

<syntaxhighlight lang="python">
# df/de = 0.5 * (1 + erf(x/sqrt(2))) + (1/sqrt(2*pi)) * x * exp(-0.5 * x^2)
t = 0.3989422804014327  # 1/sqrt(2*pi)
f_partial = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
df_de = f_partial + t * e_row * tl.exp(-0.5 * e_row * e_row)
</syntaxhighlight>

=== Long Indexing Support ===
For sequences with more than 2^31 elements, the kernel automatically switches to 64-bit indexing:

<syntaxhighlight lang="python">
INT32_SAFETY_BUFFER = 2**31 - BLOCK_SIZE * 4

# In kernel:
if LONG_INDEXING:
    offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    n_elements = tl.cast(n_elements, tl.int64)
else:
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
</syntaxhighlight>

=== Memory Efficiency ===
The backward kernel stores results in-place to minimize memory allocations:

<syntaxhighlight lang="python">
# Store derivatives in input buffers (no new allocations)
tl.store(DW + offsets, h_row, mask=mask)   # h  = f * g
tl.store(e + offsets, df_row, mask=mask)   # df = DW * f
tl.store(g + offsets, de_row, mask=mask)   # de = gradient for gate
</syntaxhighlight>

== Performance Characteristics ==

{| class="wikitable"
|+ Kernel Performance
|-
! Variant !! Forward Speed !! Backward Speed !! Numerical Precision
|-
| Exact || Baseline || Baseline || Maximum (uses erf)
|-
| Approximate || ~10% faster || ~10% faster || Slightly lower (uses tanh approximation)
|}

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_11]]
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Triton_Optimization]]
* [[related::Implementation:Unslothai_Unsloth_SwiGLU_Kernel]]
* [[related::Implementation:Unslothai_Unsloth_LayerNorm_Kernel]]

== See Also ==
* [https://arxiv.org/abs/2002.05202 GLU Variants Improve Transformer (Shazeer 2020)]
* [https://arxiv.org/abs/2305.12073 GEGLU Derivative Derivation]
* [https://www.desmos.com/calculator/nqprfoni6x GEGLU Visualization]
