# Implementation: LayerNorm_Kernel

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Kernels]], [[domain::GPU_Optimization]], [[domain::Normalization]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==
Provides a Triton-optimized LayerNorm kernel implementation with custom forward and backward passes, designed as a drop-in replacement for PyTorch's native LayerNorm.

=== Description ===
The LayerNorm Kernel module implements an efficient Triton-based Layer Normalization operation following the standard formula:

<code>y = (x - mean) / sqrt(var + eps) * weight + bias</code>

Key features:
* '''Float32 Computation''': All intermediate calculations performed in float32 for numerical stability, following PyTorch's Fp32LayerNorm pattern
* '''Cached Statistics''': Mean and inverse variance are cached during forward pass for efficient backward computation
* '''Row-wise Processing''': Each Triton program handles one row (feature dimension) for optimal memory access
* '''Automatic Block Size Tuning''': Uses utility function to calculate optimal block size and warp count based on feature dimension
* '''Gradient Computation''': Implements efficient backward pass following Andrej Karpathy's llm.c documentation

The implementation is derived from the llm.c project and optimized for transformer architectures where LayerNorm is applied frequently.

=== Usage ===
This kernel is used as a faster replacement for PyTorch's <code>nn.LayerNorm</code> when:
* Training transformer models with Unsloth optimizations
* The layer has <code>elementwise_affine=True</code> (weight and bias parameters)
* GPU memory and computation efficiency are priorities

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' unsloth/kernels/layernorm.py
* '''Lines:''' 1-225

=== Signature ===
<syntaxhighlight lang="python">
def fast_layernorm(layernorm, X):
    """
    Apply fast Triton LayerNorm to input tensor using existing LayerNorm parameters.

    Args:
        layernorm: PyTorch LayerNorm module (must have elementwise_affine=True)
        X: Input tensor of shape (..., normalized_shape)

    Returns:
        Normalized output tensor of same shape as X
    """

class Fast_Layernorm(torch.autograd.Function):
    """
    Autograd function for Triton LayerNorm with custom backward pass.
    """

    @staticmethod
    def forward(ctx, X, W, b, eps):
        """
        Args:
            ctx: Autograd context for saving tensors
            X: Input tensor
            W: Weight (gamma) parameter
            b: Bias (beta) parameter
            eps: Epsilon for numerical stability

        Returns:
            Normalized output tensor
        """

    @staticmethod
    def backward(ctx, dY):
        """
        Args:
            ctx: Autograd context with saved tensors
            dY: Upstream gradient

        Returns:
            Tuple of (dX, None, None, None) - only input gradient computed
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.layernorm import (
    fast_layernorm,
    Fast_Layernorm,
)
</syntaxhighlight>

== I/O Contract ==

=== fast_layernorm ===

{| class="wikitable"
|+ Input Parameters
|-
! Name !! Type !! Description
|-
| layernorm || nn.LayerNorm || PyTorch LayerNorm module with elementwise_affine=True
|-
| X || torch.Tensor || Input tensor of shape (..., normalized_shape)
|}

{| class="wikitable"
|+ Output
|-
! Name !! Type !! Shape !! Description
|-
| output || torch.Tensor || Same as X || Layer-normalized tensor
|}

=== Fast_Layernorm.forward ===

{| class="wikitable"
|+ Input Parameters
|-
! Name !! Type !! Shape !! Description
|-
| X || torch.Tensor || (..., n_cols) || Input tensor (flattened to 2D internally)
|-
| W || torch.Tensor || (n_cols,) || Weight (gamma) parameter
|-
| b || torch.Tensor || (n_cols,) || Bias (beta) parameter
|-
| eps || float || - || Epsilon for numerical stability (typically 1e-5 or 1e-6)
|}

{| class="wikitable"
|+ Output
|-
! Name !! Type !! Shape !! Description
|-
| Y || torch.Tensor || Same as X || Normalized output
|}

=== Saved Tensors for Backward ===

{| class="wikitable"
|+ Cached Tensors
|-
! Name !! Type !! Shape !! Description
|-
| X || torch.Tensor || (n_rows, n_cols) || Original input (flattened)
|-
| W || torch.Tensor || (n_cols,) || Weight parameter
|-
| b || torch.Tensor || (n_cols,) || Bias parameter
|-
| r || torch.Tensor || (n_rows,) || Inverse variance: 1/sqrt(var + eps)
|-
| mu || torch.Tensor || (n_rows,) || Mean per row
|}

== Usage Examples ==

=== Basic Usage with Existing LayerNorm ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from unsloth.kernels.layernorm import fast_layernorm

# Create standard PyTorch LayerNorm
hidden_size = 4096
layernorm = nn.LayerNorm(hidden_size, eps=1e-5, device="cuda", dtype=torch.bfloat16)

# Input tensor
batch, seq_len = 2, 128
X = torch.randn(batch, seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)

# Apply fast LayerNorm (drop-in replacement)
output = fast_layernorm(layernorm, X)
print(f"Output shape: {output.shape}")  # (2, 128, 4096)
</syntaxhighlight>

=== Direct Usage with Fast_Layernorm ===
<syntaxhighlight lang="python">
import torch
from unsloth.kernels.layernorm import Fast_Layernorm

hidden_size = 4096
batch, seq_len = 2, 128

# Create parameters
W = torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)
b = torch.zeros(hidden_size, device="cuda", dtype=torch.bfloat16)
eps = 1e-5

# Input tensor
X = torch.randn(batch, seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)

# Apply LayerNorm
output = Fast_Layernorm.apply(X, W, b, eps)
</syntaxhighlight>

=== Training with Gradient Computation ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from unsloth.kernels.layernorm import fast_layernorm

hidden_size = 4096
layernorm = nn.LayerNorm(hidden_size, eps=1e-5, device="cuda", dtype=torch.float16)

# Initialize with non-trivial weights for testing
torch.nn.init.uniform_(layernorm.weight)
torch.nn.init.uniform_(layernorm.bias)

# Input with gradient tracking
X = torch.randn(2, 128, hidden_size, device="cuda", dtype=torch.float16, requires_grad=True)

# Forward pass
output = fast_layernorm(layernorm, X)

# Backward pass
upstream_grad = torch.randn_like(output)
output.backward(upstream_grad)

print(f"Input gradient shape: {X.grad.shape}")  # (2, 128, 4096)
</syntaxhighlight>

=== Comparison with PyTorch LayerNorm ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from unsloth.kernels.layernorm import fast_layernorm

hidden_size = 1024
layernorm = nn.LayerNorm(hidden_size, eps=1e-5, device="cuda", dtype=torch.float16)

X = torch.randn(21, 3341, hidden_size, device="cuda", dtype=torch.float16)

# PyTorch native
with torch.no_grad():
    pytorch_output = layernorm(X)

# Unsloth Triton
with torch.no_grad():
    triton_output = fast_layernorm(layernorm, X)

# Verify numerical equivalence
print(f"Max difference: {torch.max(torch.abs(pytorch_output - triton_output))}")
</syntaxhighlight>

== Implementation Details ==

=== Forward Kernel ===
<syntaxhighlight lang="python">
@triton.jit
def layernorm_forward(Y, Y_row_stride, X, X_row_stride, W, b, r, mu,
                      n_cols, eps, BLOCK_SIZE):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row data in float32 for precision
    X_row = tl.load(X + row_idx * X_row_stride + col_offsets, mask=mask).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask).to(tl.float32)

    # Compute statistics
    mean_X = tl.sum(X_row, axis=0) / n_cols
    XX = tl.where(mask, X_row - mean_X, 0)
    row_var = tl.sum(XX * XX, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)

    # Store statistics for backward
    tl.store(r + row_idx, inv_var)
    tl.store(mu + row_idx, mean_X)

    # Compute output
    output = (XX * inv_var) * W_row + b_row
    tl.store(Y + row_idx * Y_row_stride + col_offsets, output, mask=mask)
</syntaxhighlight>

=== Backward Kernel ===
The backward pass follows the derivation from [https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md Karpathy's llm.c]:

<syntaxhighlight lang="python">
@triton.jit
def layernorm_backward(dY, dY_row_stride, X, X_row_stride, W, b, r, mu,
                       n_cols, eps, BLOCK_SIZE):
    row_idx = tl.program_id(0)

    # Load saved values
    inv_var = tl.load(r + row_idx).to(tl.float32)
    mean = tl.load(mu + row_idx).to(tl.float32)

    # Compute normalized input
    normed = (X_row - mean) * inv_var

    # Compute gradient
    dY_W = dY_row * W_row
    dX_row = (
        dY_W
        - tl.sum(dY_W, axis=0) / n_cols
        - normed * tl.sum(dY_W * normed, axis=0) / n_cols
    ) * inv_var

    tl.store(dY + row_idx * dY_row_stride + col_offsets, dX_row, mask=mask)
</syntaxhighlight>

=== Block Size Selection ===
The kernel uses the <code>calculate_settings</code> utility to determine optimal block size and warp count:

<syntaxhighlight lang="python">
from unsloth.kernels.utils import calculate_settings

n_cols = 4096
BLOCK_SIZE, num_warps = calculate_settings(n_cols)
# Returns power-of-2 BLOCK_SIZE >= n_cols, appropriate num_warps
</syntaxhighlight>

=== Numerical Precision ===
Following PyTorch's Fp32LayerNorm pattern, all computations are done in float32:

<syntaxhighlight lang="python">
# Load in float32 regardless of input dtype
X_row = tl.load(X + col_offsets, mask=mask).to(tl.float32)
W_row = tl.load(W + col_offsets, mask=mask).to(tl.float32)
b_row = tl.load(b + col_offsets, mask=mask).to(tl.float32)

# Output is stored in original dtype (automatic conversion)
</syntaxhighlight>

== Testing ==

The module includes built-in test functions:

<syntaxhighlight lang="python">
from unsloth.kernels.layernorm import test_layernorm, testing_suite_layernorm

# Single test
test_layernorm(dim=1024, eps=1e-5, dtype=torch.float16, bsz=21, seqlen=3341)

# Full test suite covering multiple configurations
testing_suite_layernorm()
# Tests: dim=[512,1024,2048], dtype=[float16,bfloat16], seqlen=[349,2048,3341]
</syntaxhighlight>

== Performance Characteristics ==

{| class="wikitable"
|+ Performance Notes
|-
! Aspect !! Description
|-
| Memory || Caches mean and inv_var per row (2 * n_rows floats)
|-
| Parallelism || One Triton program per row (n_rows parallel executions)
|-
| Precision || Float32 intermediate computation
|-
| Compatibility || Requires elementwise_affine=True
|}

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_11]]
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Triton_Optimization]]
* [[related::Implementation:Unslothai_Unsloth_RMSNorm_Kernel]]
* [[related::Implementation:Unslothai_Unsloth_GEGLU_Kernels]]
* [[related::Implementation:Unslothai_Unsloth_SwiGLU_Kernel]]

== See Also ==
* [https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md LayerNorm Backward Derivation (llm.c)]
* [https://pytorch.org/torchtune/stable/_modules/torchtune/modules/layer_norm.html PyTorch Fp32LayerNorm]
* [https://arxiv.org/abs/1607.06450 Layer Normalization Paper (Ba et al. 2016)]
