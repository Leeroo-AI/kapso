# Implementation: FP8_Kernels

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
Provides FP8 (8-bit floating point) quantization and matrix multiplication kernels optimized for DeepSeek-style models with block-wise quantization support.

=== Description ===
The FP8 Kernels module implements efficient 8-bit floating point operations for quantized model inference and training. It supports both row-wise and block-wise quantization schemes, with multiple backend implementations:

# '''FBGEMM Backend''': Preferred for datacenter GPUs (H100, A100) with fbgemm_gpu >= 1.4.0
# '''TorchAO Backend''': Fallback using torchao's blockwise_fp8_gemm, approximately 3x faster than pure Triton
# '''Triton Backend''': Pure Triton implementation for maximum compatibility

Key features:
* Block-wise FP8 quantization with configurable block sizes (default 128x128)
* Row-wise FP8 quantization for simpler models
* Automatic weight dequantization during backward pass for gradient computation
* Dynamic activation quantization with per-block scaling
* Automatic backend selection based on GPU capabilities and library availability

The module handles edge cases like:
* Transposed weights during backward pass
* Non-divisible-by-8 dimensions requiring dequantization fallback
* Consumer GPUs (RTX 4090, 5090) that may not support FBGEMM kernels

=== Usage ===
This kernel is used when:
* Loading DeepSeek V3 or similar FP8-quantized models
* Training with FP8 precision for memory efficiency
* Performing inference with block-quantized weights
* Working with models using <code>FP8Linear</code> or <code>FbgemmFp8Linear</code> layers

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' unsloth/kernels/fp8.py
* '''Lines:''' 1-615

=== Signature ===
<syntaxhighlight lang="python">
def fp8_linear(X, weight, weight_scale, bias=None):
    """
    Main entry point for FP8 linear operations. Automatically selects
    between block-quantized and row-quantized implementations.

    Args:
        X: Input tensor of shape (..., in_features)
        weight: FP8 weight tensor of shape (out_features, in_features)
        weight_scale: Scale tensor for weight dequantization
        bias: Optional bias tensor

    Returns:
        Output tensor of shape (..., out_features)
    """

def act_quant(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activations to FP8 with block-wise scaling.

    Args:
        x: Input tensor to quantize
        block_size: Block size for quantization (default: 128)

    Returns:
        Tuple of (quantized_tensor, scale_tensor)
    """

def weight_dequant(x: torch.Tensor, s: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """
    Dequantize FP8 weights back to higher precision.

    Args:
        x: FP8 weight tensor
        s: Scale tensor
        dtype: Output dtype (default: torch.bfloat16)

    Returns:
        Dequantized weight tensor
    """

class FP8BlockQuantLinear(torch.autograd.Function):
    """
    Autograd function for block-quantized FP8 linear operations.
    Forward: FP8 matmul with block scaling
    Backward: Dequantize weights for gradient computation
    """

class FbgemmFp8Linear_matmul(torch.autograd.Function):
    """
    Autograd function using FBGEMM's row-wise FP8 operations.
    Optimized for shapes divisible by 8.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.fp8 import (
    fp8_linear,
    act_quant,
    weight_dequant,
    FP8BlockQuantLinear,
    FbgemmFp8Linear_matmul,
    fp8_block_quant_linear,
    fbgemm_fp8_linear,
)
</syntaxhighlight>

== I/O Contract ==

=== fp8_linear ===

{| class="wikitable"
|+ Input Parameters
|-
! Name !! Type !! Shape !! Description
|-
| X || torch.Tensor || (..., in_features) || Input activation tensor
|-
| weight || torch.Tensor (float8_e4m3fn) || (out_features, in_features) || FP8 quantized weight
|-
| weight_scale || torch.Tensor || (scale_rows, scale_cols) || Per-block or per-row scale factors
|-
| bias || torch.Tensor or None || (out_features,) || Optional bias tensor
|}

{| class="wikitable"
|+ Output
|-
! Name !! Type !! Shape !! Description
|-
| output || torch.Tensor || (..., out_features) || Linear transformation result
|}

=== act_quant ===

{| class="wikitable"
|+ Input Parameters
|-
! Name !! Type !! Default !! Description
|-
| x || torch.Tensor || - || Input tensor to quantize (must have last dim divisible by block_size)
|-
| block_size || int || 128 || Quantization block size
|}

{| class="wikitable"
|+ Output
|-
! Name !! Type !! Description
|-
| y || torch.Tensor (float8_e4m3fn) || Quantized tensor
|-
| s || torch.Tensor (float32) || Scale factors of shape (..., num_blocks)
|}

=== weight_dequant ===

{| class="wikitable"
|+ Input Parameters
|-
! Name !! Type !! Description
|-
| x || torch.Tensor || FP8 weight tensor
|-
| s || torch.Tensor || Scale tensor (1D for row-wise, 2D for block-wise)
|-
| dtype || torch.dtype || Output dtype (default: torch.bfloat16)
|}

{| class="wikitable"
|+ Output
|-
! Name !! Type !! Description
|-
| y || torch.Tensor || Dequantized weight in specified dtype
|}

== Usage Examples ==

=== Basic FP8 Linear Operation ===
<syntaxhighlight lang="python">
import torch
from unsloth.kernels.fp8 import fp8_linear

# Simulated FP8 weight and scale
in_features, out_features = 4096, 4096
batch_size, seq_len = 2, 128

# Create FP8 weight (normally loaded from model)
weight = torch.randn(out_features, in_features, device="cuda").to(torch.float8_e4m3fn)

# Block-wise scale: (out_features/128, in_features/128) for 128x128 blocks
weight_scale = torch.ones(out_features // 128, in_features // 128,
                          device="cuda", dtype=torch.float32)

# Input activations
X = torch.randn(batch_size, seq_len, in_features, device="cuda", dtype=torch.bfloat16)

# Perform FP8 linear
output = fp8_linear(X, weight, weight_scale)
print(f"Output shape: {output.shape}")  # (2, 128, 4096)
</syntaxhighlight>

=== Activation Quantization ===
<syntaxhighlight lang="python">
from unsloth.kernels.fp8 import act_quant

# Quantize activations for FP8 matmul
X = torch.randn(2, 128, 4096, device="cuda", dtype=torch.bfloat16)

# Quantize with block size 128
X_quant, X_scale = act_quant(X, block_size=128)

print(f"Quantized dtype: {X_quant.dtype}")  # torch.float8_e4m3fn
print(f"Scale shape: {X_scale.shape}")  # (2, 128, 32) for 4096/128=32 blocks
</syntaxhighlight>

=== Weight Dequantization ===
<syntaxhighlight lang="python">
from unsloth.kernels.fp8 import weight_dequant

# Dequantize for gradient computation
weight_fp8 = torch.randn(4096, 4096, device="cuda").to(torch.float8_e4m3fn)
weight_scale = torch.ones(32, 32, device="cuda", dtype=torch.float32)

# Dequantize back to bfloat16
weight_bf16 = weight_dequant(weight_fp8, weight_scale, dtype=torch.bfloat16)
print(f"Dequantized dtype: {weight_bf16.dtype}")  # torch.bfloat16
</syntaxhighlight>

=== Training with FP8 ===
<syntaxhighlight lang="python">
import torch
from unsloth.kernels.fp8 import FP8BlockQuantLinear

# Enable gradients for training
X = torch.randn(2, 128, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
weight = torch.randn(4096, 4096, device="cuda").to(torch.float8_e4m3fn)
weight_scale = torch.ones(32, 32, device="cuda", dtype=torch.float32)
weight_scale.block_size = [128, 128]

# Forward pass with autograd support
output = FP8BlockQuantLinear.apply(X, weight, weight_scale)

# Backward pass works - weights are dequantized for gradient computation
loss = output.sum()
loss.backward()

print(f"Input gradient shape: {X.grad.shape}")  # (2, 128, 4096)
</syntaxhighlight>

== Implementation Details ==

=== Block-wise Quantization ===
Block-wise FP8 quantization divides the weight matrix into blocks (typically 128x128) and computes a separate scale factor for each block:

<syntaxhighlight lang="python">
# For a weight matrix of shape (M, N) with block_size (bm, bn):
# - Quantized weight: shape (M, N), dtype float8_e4m3fn
# - Scale: shape (M/bm, N/bn), dtype float32

# Quantization: y = clamp(x / scale, -448, 448)
# Dequantization: x = y * scale
</syntaxhighlight>

=== Triton Kernels ===

'''Activation Quantization Kernel:'''
<syntaxhighlight lang="python">
@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0  # FP8 E4M3 max value
    s = 1.0 if s == 0 else s  # Handle zero blocks
    y = (x / s).to(tl.float8e4nv)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)
</syntaxhighlight>

'''Weight Dequantization Kernel:'''
<syntaxhighlight lang="python">
@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # Load FP8 block and scale, multiply to dequantize
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)
</syntaxhighlight>

=== Backend Selection ===
The module automatically selects the best available backend:

<syntaxhighlight lang="python">
# Priority order:
# 1. FBGEMM >= 1.4.0 (if GPU supports CUTLASS SM90 kernels)
# 2. TorchAO blockwise_fp8_gemm (3x faster than Triton)
# 3. Pure Triton w8a8_block_fp8_matmul (maximum compatibility)

# Environment variable for manual control:
os.environ["UNSLOTH_HAS_FBGEMM"] = "1"  # Force FBGEMM
os.environ["UNSLOTH_HAS_FBGEMM"] = "0"  # Disable FBGEMM
</syntaxhighlight>

=== GPU Compatibility ===
FBGEMM kernels may not work on:
* Consumer GPUs (RTX 4090, RTX 5090)
* Blackwell architecture (B100, B200) with SM100

The module tests FBGEMM compatibility at import time and falls back to TorchAO/Triton if needed.

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_11]]
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Triton_Optimization]]
* [[related::Implementation:Unslothai_Unsloth_GEGLU_Kernels]]
* [[related::Implementation:Unslothai_Unsloth_LayerNorm_Kernel]]

== See Also ==
* [https://github.com/deepseek-ai/DeepSeek-V3 DeepSeek V3 Inference Kernels]
* [https://github.com/pytorch/FBGEMM FBGEMM GPU Library]
* [https://github.com/pytorch/ao TorchAO Quantization Library]
