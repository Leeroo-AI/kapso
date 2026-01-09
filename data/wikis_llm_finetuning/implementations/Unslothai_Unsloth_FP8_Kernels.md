# Implementation: FP8_Kernels

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|DeepSeek-V3|https://arxiv.org/abs/2412.19437]]
|-
! Domains
| [[domain::Kernels]], [[domain::Quantization]], [[domain::FP8]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Triton and FBGEMM-based kernels for FP8 (8-bit floating point) matrix multiplication with block-wise and row-wise quantization support.

=== Description ===
This module provides optimized FP8 matmul implementations for training and inference with quantized models (e.g., DeepSeek-V3). It includes Triton kernels for block-wise FP8 quantization/dequantization, autograd-compatible linear layers, and automatic backend selection (FBGEMM > TorchAO > Triton) based on availability and GPU compatibility.

=== Usage ===
Use these functions when working with FP8-quantized models like DeepSeek-V3. The module automatically selects the fastest available backend for your GPU.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fp8.py unsloth/kernels/fp8.py]
* '''Lines:''' 1-615

=== Key Functions ===
<syntaxhighlight lang="python">
def weight_dequant(
    x: torch.Tensor,      # FP8 weight tensor
    s: torch.Tensor,      # Scale tensor
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 weights to higher precision."""

def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to FP8 with per-block scales."""

def w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,      # FP8 activations
    B: torch.Tensor,      # FP8 weights
    As: torch.Tensor,     # Activation scales
    Bs: torch.Tensor,     # Weight scales
    block_size: List[int],  # [block_n, block_k]
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Block-wise FP8 matmul using Triton kernels."""

def fp8_linear(
    X: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unified FP8 linear layer (auto-selects block/row mode)."""

class FP8BlockQuantLinear(torch.autograd.Function):
    """Autograd function for block-quantized FP8 linear with backward support."""

class FbgemmFp8Linear_matmul(torch.autograd.Function):
    """Autograd function for FBGEMM row-quantized FP8 linear."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.fp8 import (
    fp8_linear,
    weight_dequant,
    act_quant,
    w8a8_block_fp8_matmul_triton,
    fp8_block_quant_linear,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs (fp8_linear) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| X || Tensor || Yes || Input activations [batch, seq_len, hidden_dim]
|-
| weight || Tensor || Yes || FP8 weight tensor [out_features, in_features]
|-
| weight_scale || Tensor || Yes || Scale tensor (2D for block, 1D for row quantization)
|-
| bias || Tensor || No || Optional bias tensor
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output || Tensor || Linear layer output in X.dtype
|}

== Backend Selection ==

The module automatically selects the best FP8 backend:

{| class="wikitable"
|-
! Priority !! Backend !! Condition !! Notes
|-
| 1 || FBGEMM f8f8bf16_blockwise || fbgemm_gpu >= 1.4.0, test_has_fbgemm() passes || 15% faster than TorchAO
|-
| 2 || TorchAO blockwise_fp8_gemm || torchao available || 3x faster than pure Triton
|-
| 3 || Triton w8a8_block_fp8_matmul || Always available || Fallback implementation
|}

FBGEMM is automatically disabled on:
- Consumer GPUs (RTX 4090/5090) - CUTLASS kernel failures
- SM100 Blackwell GPUs - Architecture mismatch errors
- fbgemm_gpu < 1.4.0 - Numerical precision issues

== Usage Examples ==

=== Basic FP8 Linear ===
<syntaxhighlight lang="python">
from unsloth.kernels.fp8 import fp8_linear
import torch

# Assume FP8 weights from a quantized model
# weight: [out_features, in_features] in torch.float8_e4m3fn
# weight_scale: [out_features // 128, in_features // 128] for block quant
# or [out_features, 1] for row quant

x = torch.randn(2, 512, 4096, dtype=torch.bfloat16, device="cuda")
output = fp8_linear(x, weight, weight_scale)
# output: [2, 512, out_features] in bfloat16
</syntaxhighlight>

=== Manual Activation Quantization ===
<syntaxhighlight lang="python">
from unsloth.kernels.fp8 import act_quant, w8a8_block_fp8_matmul_triton

# Quantize activations
x_fp8, x_scale = act_quant(x, block_size=128)

# Perform FP8 matmul
output = w8a8_block_fp8_matmul_triton(
    x_fp8, weight, x_scale, weight_scale,
    block_size=[128, 128],
    output_dtype=torch.bfloat16,
)
</syntaxhighlight>

=== Dequantize Weights ===
<syntaxhighlight lang="python">
from unsloth.kernels.fp8 import weight_dequant

# Dequantize for debugging or non-FP8 operations
weight_bf16 = weight_dequant(weight_fp8, weight_scale, dtype=torch.bfloat16)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
