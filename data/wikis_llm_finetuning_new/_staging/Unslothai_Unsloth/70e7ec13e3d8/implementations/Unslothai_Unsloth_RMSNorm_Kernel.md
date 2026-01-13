# Implementation: RMSNorm Kernel

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

The RMSNorm (Root Mean Square Layer Normalization) kernel provides a high-performance Triton implementation of RMS normalization, a simplified alternative to LayerNorm that has become standard in modern LLMs like LLaMA and Gemma. This implementation includes both forward and backward pass kernels with special handling for Gemma-style normalization.

RMS LayerNorm normalizes by the root mean square of activations rather than mean and variance, reducing computational overhead while maintaining model quality. The formula is:

<code>y = x * rsqrt(mean(x^2) + eps) * weight</code>

For Gemma models, the weight is applied as <code>(weight + 1.0)</code> to match the original implementation.

== Code Reference ==

'''File:''' <code>unsloth/kernels/rms_layernorm.py</code>

=== Forward Kernel ===

<syntaxhighlight lang="python">
@triton.jit
def _rms_layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fast RMS Layernorm kernel"""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask=mask)
</syntaxhighlight>

=== Gemma-Style Forward Kernel ===

<syntaxhighlight lang="python">
@triton.jit
def _gemma_rms_layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Applies (weight + 1.0) normalization for Gemma compatibility
    ...
    output = normed * (W_row + 1.0)
    tl.store(Y + col_offsets, output, mask=mask)
</syntaxhighlight>

=== Backward Kernel ===

<syntaxhighlight lang="python">
def _rms_layernorm_backward(
    dY,
    dY_row_stride: tl.constexpr,
    dX,
    dX_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fast RMS Layernorm backward pass kernel"""
    ...
</syntaxhighlight>

=== PyTorch Autograd Function ===

<syntaxhighlight lang="python">
class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, eps: float, gemma: bool = False):
        shape = X.shape
        dim: int = shape[-1]
        X = X.reshape(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        ...
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        ...
        return dX, None, None, None
</syntaxhighlight>

== I/O Contract ==

=== fast_rms_layernorm ===

'''Signature:'''
<syntaxhighlight lang="python">
def fast_rms_layernorm(layernorm, X: torch.Tensor, gemma: bool = False) -> torch.Tensor
</syntaxhighlight>

'''Inputs:'''
{| class="wikitable"
|-
! Parameter !! Type !! Description
|-
| <code>layernorm</code> || <code>nn.Module</code> || RMSNorm layer with <code>weight</code> and <code>variance_epsilon</code>/<code>eps</code> attributes
|-
| <code>X</code> || <code>torch.Tensor</code> || Input tensor of shape <code>(batch, seq_len, dim)</code> or <code>(batch*seq_len, dim)</code>
|-
| <code>gemma</code> || <code>bool</code> || Use Gemma-style normalization (<code>weight + 1.0</code>)
|}

'''Outputs:'''
{| class="wikitable"
|-
! Return !! Type !! Description
|-
| <code>out</code> || <code>torch.Tensor</code> || Normalized output tensor, same shape as input
|}

'''Constraints:'''
* Hidden dimension must not exceed <code>MAX_FUSED_SIZE</code> (65536)
* Input must be contiguous or will be reshaped
* Computation performed in float32, output cast back to input dtype

=== Unsloth_LlamaRMSNorm ===

Drop-in replacement class for <code>transformers.models.llama.modeling_llama.LlamaRMSNorm</code>:

<syntaxhighlight lang="python">
class Unsloth_LlamaRMSNorm(LlamaRMSNorm):
    def forward(self, X):
        return fast_rms_layernorm(self, X, gemma=False)
</syntaxhighlight>

== Usage Examples ==

=== Direct Kernel Usage ===

<syntaxhighlight lang="python">
from unsloth.kernels import fast_rms_layernorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# Create layernorm
layernorm = LlamaRMSNorm((4096,), eps=1e-5).cuda()

# Input tensor
X = torch.randn(2, 1024, 4096, dtype=torch.float16, device="cuda")

# Apply fast RMS normalization
out = fast_rms_layernorm(layernorm, X, gemma=False)

# For Gemma models
out_gemma = fast_rms_layernorm(layernorm, X, gemma=True)
</syntaxhighlight>

=== Model Patching ===

<syntaxhighlight lang="python">
from unsloth.kernels.rms_layernorm import patch_rms_layernorm, unpatch_rms_layernorm

# Patch transformers to use fast kernels
patch_rms_layernorm()

# ... use LLaMA or Mllama models with automatic fast normalization ...

# Restore original behavior
unpatch_rms_layernorm()
</syntaxhighlight>

=== With Gradient Computation ===

<syntaxhighlight lang="python">
X = torch.randn(2, 1024, 4096, dtype=torch.float16, device="cuda", requires_grad=True)
layernorm = LlamaRMSNorm((4096,), eps=1e-5).cuda()

# Forward pass
out = fast_rms_layernorm(layernorm, X)

# Backward pass (gradients computed via custom backward kernel)
loss = out.sum()
loss.backward()  # Uses _rms_layernorm_backward kernel

print(X.grad.shape)  # (2, 1024, 4096)
</syntaxhighlight>

== Related Pages ==

* [[Unslothai_Unsloth_Kernel_Utils]] - Utility functions including <code>calculate_settings</code> and <code>torch_gpu_device</code>
* [[Unslothai_Unsloth_SwiGLU_Kernel]] - Another Triton kernel for activation functions
* [[Unslothai_Unsloth_RoPE_Kernel]] - Rotary position embedding kernel
