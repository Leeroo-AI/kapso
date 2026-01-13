# Implementation: Kernel Utils

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Kernels]], [[domain::GPU_Optimization]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

The kernel utilities module provides foundational functions and constants used across all Unsloth Triton kernels. It includes device detection, kernel configuration calculation, CUDA/XPU stream management, quantized weight handling, and optimized linear operations for both training and inference.

Key components:
* '''calculate_settings''' - Compute optimal block size and warp count for Triton kernels
* '''torch_gpu_device''' - Context manager for multi-GPU device selection
* '''QUANT_STATE''' - Access quantization state from bitsandbytes weights
* '''triton_tanh''' - Version-compatible Triton tanh function
* '''fast_dequantize''' - High-performance NF4/FP8 dequantization
* '''fast_gemv''' - Optimized matrix-vector multiplication for inference
* '''matmul_lora''' - Fused matrix multiplication with LoRA adaptation

== Code Reference ==

'''File:''' <code>unsloth/kernels/utils.py</code>

=== Constants ===

<syntaxhighlight lang="python">
MAX_FUSED_SIZE: int = 65536  # Maximum block size for Triton kernels
next_power_of_2 = triton.next_power_of_2
</syntaxhighlight>

=== calculate_settings ===

<syntaxhighlight lang="python">
def calculate_settings(n: int) -> (int, int):
    """
    Calculate optimal BLOCK_SIZE and num_warps for Triton kernels.

    Returns:
        BLOCK_SIZE: Next power of 2 >= n (capped at MAX_FUSED_SIZE)
        num_warps: Optimal warp count based on block size
    """
    BLOCK_SIZE: int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps: int = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps
</syntaxhighlight>

=== triton_tanh (Version-Compatible) ===

<syntaxhighlight lang="python">
from unsloth_zoo.utils import Version

if Version(triton.__version__) >= Version("3.0.0"):
    if DEVICE_TYPE == "xpu":
        triton_tanh = tl.extra.intel.libdevice.tanh
    else:
        from triton.language.extra import libdevice
        triton_tanh = libdevice.tanh
    triton_cast = tl.cast
else:
    triton_tanh = tl.math.tanh

    @triton.jit
    def triton_cast(x, dtype):
        return x.to(dtype)
</syntaxhighlight>

=== torch_gpu_device Context Manager ===

<syntaxhighlight lang="python">
if DEVICE_COUNT > 1:
    if DEVICE_TYPE in ("cuda", "hip"):
        torch_gpu_device = torch.cuda.device
    elif DEVICE_TYPE == "xpu":
        torch_gpu_device = torch.xpu.device
else:
    from contextlib import nullcontext

    def torch_gpu_device(device):
        return nullcontext()
</syntaxhighlight>

=== QUANT_STATE Helper ===

<syntaxhighlight lang="python">
def QUANT_STATE(W):
    """Get quantization state from a bitsandbytes quantized weight."""
    return getattr(W, "quant_state", None)
</syntaxhighlight>

=== get_lora_parameters ===

<syntaxhighlight lang="python">
def get_lora_parameters(proj):
    """
    Return a 5-tuple of (weight, weight quant_state, lora A, lora B, and lora scale).
    If QAT is enabled, additionally fake quantize the base layer and lora weights.
    """
    base_layer = getattr(proj, "base_layer", proj)
    W = base_layer.weight

    # Optionally apply fake quantization for QAT
    if hasattr(base_layer, "weight_fake_quantizer"):
        weight_fake_quantizer = getattr(base_layer, "weight_fake_quantizer", None)
        if weight_fake_quantizer is not None:
            W = weight_fake_quantizer(W)

    # Get quant state for 4bit or FP8
    W_quant = getattr(W, "quant_state", None)
    if W_quant is None:
        W_quant = getattr(base_layer, "weight_scale_inv", None)
        if W_quant is None:
            W_quant = getattr(base_layer, "weight_scale", None)

    # Handle disabled/merged adapters
    if getattr(proj, "disable_adapters", True) or proj.merged:
        return W, W_quant, None, None, None

    # Get active adapter weights
    adapter = getattr(proj, "active_adapters", None)
    if adapter is None:
        adapter = getattr(proj, "active_adapter", ("default"))
    adapter = adapter[0]

    return (W, W_quant, proj.lora_A[adapter].weight,
            proj.lora_B[adapter].weight, proj.scaling[adapter])
</syntaxhighlight>

=== fast_dequantize ===

<syntaxhighlight lang="python">
@torch.inference_mode
def fast_dequantize(W, quant_state=None, out=None, use_global_buffer=False):
    """
    High-performance dequantization for NF4 and FP8 weights.

    Supports:
    - Float8Tensor (torchao)
    - FP8 (torch.float8_e4m3fn)
    - NF4 (bitsandbytes)

    Args:
        W: Quantized weight tensor
        quant_state: Quantization state (absmax, shape, dtype, blocksize, etc.)
        out: Optional pre-allocated output buffer
        use_global_buffer: Use shared global buffers for inference efficiency

    Returns:
        Dequantized weight tensor
    """
    if isinstance(W, Float8Tensor):
        return W.dequantize()
    if quant_state is None:
        return W
    if W.dtype == torch.float8_e4m3fn:
        return weight_dequant(W, quant_state)

    # NF4 dequantization with CUDA stream support
    ...
</syntaxhighlight>

=== fast_gemv (Inference Optimized) ===

<syntaxhighlight lang="python">
def fast_gemv(X, W, quant_state, out=None):
    """
    Fast X @ W for sequence length == 1 inference.

    Optimized path using bitsandbytes' 4-bit GEMV kernel with
    CUDA stream support for maximum throughput.
    """
    if quant_state is None:
        return torch_matmul(X, W, out=out)

    _, q_len, hd = X.shape
    # Uses cgemm_4bit_inference_naive_fp16/bf16 for quantized inference
    ...
</syntaxhighlight>

=== matmul_lora ===

<syntaxhighlight lang="python">
def matmul_lora(X, W, W_quant, A, B, s, out=None):
    """
    Fused matrix multiplication with LoRA: X @ W.T + s * (X @ A.T @ B.T)

    Handles:
    - Float8Tensor weights
    - FP8 weights (torch.float8_e4m3fn)
    - NF4 quantized weights
    - Standard float weights

    Args:
        X: Input tensor (batch, seq_len, in_dim) or (batch*seq_len, in_dim)
        W: Weight matrix (out_dim, in_dim) - possibly quantized
        W_quant: Quantization state or scale
        A: LoRA A matrix (rank, in_dim) or None
        B: LoRA B matrix (out_dim, rank) or None
        s: LoRA scaling factor or None

    Returns:
        Output tensor with LoRA applied
    """
    dtype = X.dtype

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    # Handle different weight formats
    if isinstance(W, Float8Tensor):
        ...
    elif W.dtype == torch.float8_e4m3fn:
        out = fp8_linear(X, W, W_quant)
    else:
        W = fast_dequantize(W, W_quant, use_global_buffer=True)
        out = torch_matmul(X, W.t(), out=out)

    # Add LoRA contribution
    if A is not None:
        A, B = A.t(), B.t()
        XA = torch_matmul(X, A.to(dtype))
        out.addmm_(XA, B.to(dtype), alpha=s)

    return out.view(batch, seq_len, -1) if reshape else out
</syntaxhighlight>

== I/O Contract ==

=== calculate_settings ===

'''Signature:'''
<syntaxhighlight lang="python">
def calculate_settings(n: int) -> Tuple[int, int]
</syntaxhighlight>

'''Inputs:'''
{| class="wikitable"
|-
! Parameter !! Type !! Description
|-
| <code>n</code> || <code>int</code> || Target dimension (e.g., hidden_dim, head_dim)
|}

'''Outputs:'''
{| class="wikitable"
|-
! Return !! Type !! Description
|-
| <code>BLOCK_SIZE</code> || <code>int</code> || Optimal block size (power of 2)
|-
| <code>num_warps</code> || <code>int</code> || Optimal warp count (4, 8, 16, or 32)
|}

'''Warp Count Mapping:'''
{| class="wikitable"
|-
! BLOCK_SIZE Range !! num_warps
|-
| < 2048 || 4
|-
| 2048 - 8191 || 8
|-
| 8192 - 32767 || 16
|-
| >= 32768 || 32
|}

=== get_lora_parameters ===

'''Signature:'''
<syntaxhighlight lang="python">
def get_lora_parameters(proj) -> Tuple[Tensor, Optional[Any], Optional[Tensor], Optional[Tensor], Optional[float]]
</syntaxhighlight>

'''Outputs:'''
{| class="wikitable"
|-
! Index !! Type !! Description
|-
| 0 || <code>torch.Tensor</code> || Base weight W (possibly quantized)
|-
| 1 || <code>Any</code> or <code>None</code> || Quantization state
|-
| 2 || <code>torch.Tensor</code> or <code>None</code> || LoRA A weight
|-
| 3 || <code>torch.Tensor</code> or <code>None</code> || LoRA B weight
|-
| 4 || <code>float</code> or <code>None</code> || LoRA scaling factor
|}

=== fast_dequantize ===

'''Signature:'''
<syntaxhighlight lang="python">
def fast_dequantize(
    W: torch.Tensor,
    quant_state: Optional[Any] = None,
    out: Optional[torch.Tensor] = None,
    use_global_buffer: bool = False
) -> torch.Tensor
</syntaxhighlight>

'''Supported Weight Types:'''
* <code>torch.float8_e4m3fn</code> - FP8 quantization
* <code>Float8Tensor</code> - TorchAO Float8
* NF4 with double quantization (bitsandbytes)

=== matmul_lora ===

'''Signature:'''
<syntaxhighlight lang="python">
def matmul_lora(
    X: torch.Tensor,
    W: torch.Tensor,
    W_quant: Optional[Any],
    A: Optional[torch.Tensor],
    B: Optional[torch.Tensor],
    s: Optional[float],
    out: Optional[torch.Tensor] = None
) -> torch.Tensor
</syntaxhighlight>

== Usage Examples ==

=== Kernel Configuration ===

<syntaxhighlight lang="python">
from unsloth.kernels.utils import calculate_settings

# For LLaMA-7B hidden dimension
hidden_dim = 4096
BLOCK_SIZE, num_warps = calculate_settings(hidden_dim)
print(f"BLOCK_SIZE={BLOCK_SIZE}, num_warps={num_warps}")
# Output: BLOCK_SIZE=4096, num_warps=8

# For head dimension in attention
head_dim = 128
BLOCK_SIZE, num_warps = calculate_settings(head_dim)
print(f"BLOCK_SIZE={BLOCK_SIZE}, num_warps={num_warps}")
# Output: BLOCK_SIZE=128, num_warps=4
</syntaxhighlight>

=== Multi-GPU Device Context ===

<syntaxhighlight lang="python">
from unsloth.kernels.utils import torch_gpu_device

# Ensure kernel runs on correct device in multi-GPU setup
device = torch.device("cuda:1")
X = torch.randn(1024, 4096, device=device)

with torch_gpu_device(device):
    # Triton kernel launches on cuda:1
    _my_kernel[grid](X, ...)
</syntaxhighlight>

=== Dequantizing 4-bit Weights ===

<syntaxhighlight lang="python">
from unsloth.kernels.utils import fast_dequantize, QUANT_STATE
import bitsandbytes as bnb

# Load a 4-bit quantized weight
linear = bnb.nn.Linear4bit(4096, 4096, bias=False)
W = linear.weight
quant_state = QUANT_STATE(W)

# Dequantize for computation
W_fp16 = fast_dequantize(W, quant_state)
print(W_fp16.shape, W_fp16.dtype)
# Output: torch.Size([4096, 4096]) torch.float16

# Use global buffer for repeated inference (memory efficient)
W_fp16 = fast_dequantize(W, quant_state, use_global_buffer=True)
</syntaxhighlight>

=== LoRA-Aware Matrix Multiplication ===

<syntaxhighlight lang="python">
from unsloth.kernels.utils import matmul_lora, get_lora_parameters

# Get LoRA parameters from a PEFT layer
W, W_quant, A, B, s = get_lora_parameters(model.layer.self_attn.q_proj)

# Input tensor
X = torch.randn(2, 1024, 4096, dtype=torch.float16, device="cuda")

# Compute X @ W.T + s * (X @ A.T @ B.T)
out = matmul_lora(X, W, W_quant, A, B, s)
print(out.shape)  # (2, 1024, out_dim)
</syntaxhighlight>

=== Fast Inference with GEMV ===

<syntaxhighlight lang="python">
from unsloth.kernels.utils import fast_gemv, fast_linear_forward

# Single-token inference (batch=1, seq_len=1)
X = torch.randn(1, 1, 4096, dtype=torch.float16, device="cuda")

# Direct GEMV for quantized weights
out = fast_gemv(X, W, quant_state)

# Or use fast_linear_forward for complete LoRA handling
out = fast_linear_forward(model.layer.self_attn.q_proj, X)
</syntaxhighlight>

=== Using triton_tanh in Custom Kernels ===

<syntaxhighlight lang="python">
from unsloth.kernels.utils import triton_tanh

@triton.jit
def my_custom_kernel(X, Y, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets)

    # Version-compatible tanh
    y = triton_tanh(x)

    tl.store(Y + offsets, y)
</syntaxhighlight>

== Related Pages ==

* [[Unslothai_Unsloth_RMSNorm_Kernel]] - Uses <code>calculate_settings</code> and <code>torch_gpu_device</code>
* [[Unslothai_Unsloth_RoPE_Kernel]] - Uses <code>calculate_settings</code> and <code>torch_gpu_device</code>
* [[Unslothai_Unsloth_SwiGLU_Kernel]] - Uses <code>torch_gpu_device</code>
