# Implementation: FP8 Quantization Kernels

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Repo|DeepSeek-V3|https://github.com/deepseek-ai/DeepSeek-V3]]
* [[source::Doc|NVIDIA FP8 Formats|https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Quantization]], [[domain::Kernels]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
FP8 (8-bit floating point) quantization and dequantization operations for memory-efficient inference and training, supporting multiple backends including FBGEMM, TorchAO, and custom Triton kernels.

=== Description ===
The FP8 Quantization module provides comprehensive support for 8-bit floating point operations, enabling approximately 2x memory reduction compared to FP16/BF16 while maintaining acceptable accuracy. This is critical for running very large models like DeepSeek-V3 efficiently.

The implementation provides multiple approaches:

1. **Row-wise Quantization** - Simpler per-row scaling factors for basic quantization
2. **Block-wise Quantization (128x128)** - Finer granularity for better accuracy
3. **Multiple Backends**:
   - **FBGEMM** - Facebook's optimized kernels (fastest on NVIDIA H100)
   - **TorchAO** - PyTorch's native quantization library
   - **Custom Triton** - Fallback kernels for broader GPU support

Key classes and functions:
- `FP8BlockQuantLinear` - Block-quantized linear layer wrapper
- `weight_dequant()` / `weight_dequant_block()` - Dequantization kernels
- `act_quant()` - Activation quantization for inference
- `w8a8_block_fp8_matmul_triton()` - Fused FP8 matrix multiplication

The module handles numerical stability (preventing NaNs from high activation values), GPU capability detection, and automatic backend selection.

=== Usage ===
Import this module when working with FP8-quantized models (e.g., DeepSeek-V3, Llama with FP8) or when implementing memory-efficient inference pipelines. The kernels are automatically used when loading compatible models through Unsloth.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fp8.py#L1-L200 unsloth/kernels/fp8.py]
* '''Lines:''' 1-599 (full module)

=== Signature ===
<syntaxhighlight lang="python">
class FP8BlockQuantLinear(nn.Module):
    """
    Block-quantized FP8 linear layer for memory-efficient inference.

    Wraps an existing linear layer and applies block-wise FP8 quantization
    to weights and activations.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP8 quantized matmul."""

def weight_dequant(x: torch.Tensor, s: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """
    Dequantize FP8 weights to higher precision.

    Args:
        x: Quantized weight tensor in FP8 format
        s: Scale factors (per-row or per-block)
        dtype: Target dtype for dequantized output

    Returns:
        Dequantized weight tensor in specified dtype
    """

def act_quant(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activations to FP8 format.

    Args:
        x: Input activation tensor
        block_size: Block size for quantization (default: 128)

    Returns:
        Tuple of (quantized_tensor, scale_factors)
    """

def w8a8_block_fp8_matmul_triton(
    A: torch.Tensor, B: torch.Tensor,
    As: torch.Tensor, Bs: torch.Tensor,
    block_size: int = 128
) -> torch.Tensor:
    """
    FP8 matrix multiplication with block-wise quantization.

    Args:
        A: First input matrix (quantized)
        B: Second input matrix (quantized)
        As: Scale factors for A
        Bs: Scale factors for B
        block_size: Quantization block size

    Returns:
        Result of A @ B in higher precision
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.fp8 import (
    FP8BlockQuantLinear,
    weight_dequant,
    act_quant,
    w8a8_block_fp8_matmul_triton,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| x || torch.Tensor (FP8) || Yes || Quantized weight or activation tensor
|-
| s || torch.Tensor (float32) || Yes || Scale factors for dequantization (per-row or per-block)
|-
| block_size || int || No || Block size for block-wise quantization, default 128
|-
| dtype || torch.dtype || No || Target dtype for dequantization, default bfloat16
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| quantized || torch.Tensor (FP8) || Quantized tensor in float8_e4m3fn format
|-
| scales || torch.Tensor (float32) || Per-block or per-row scale factors
|-
| output || torch.Tensor || Dequantized or matmul result in target dtype
|}

== Usage Examples ==

=== Weight Dequantization ===
<syntaxhighlight lang="python">
import torch
from unsloth.kernels.fp8 import weight_dequant

# Simulating FP8 quantized weights
weight_fp8 = torch.randn(4096, 4096, device="cuda", dtype=torch.float8_e4m3fn)
# Row-wise scales (one per row)
scales_row = torch.randn(4096, 1, device="cuda", dtype=torch.float32).abs()

# Dequantize for computation
weight_bf16 = weight_dequant(weight_fp8, scales_row, dtype=torch.bfloat16)
print(f"Dequantized shape: {weight_bf16.shape}, dtype: {weight_bf16.dtype}")
</syntaxhighlight>

=== Activation Quantization ===
<syntaxhighlight lang="python">
import torch
from unsloth.kernels.fp8 import act_quant

# Input activations in BF16
activations = torch.randn(2, 512, 4096, device="cuda", dtype=torch.bfloat16)

# Quantize to FP8 for efficient matmul
act_fp8, scales = act_quant(activations.view(-1, 128), block_size=128)
print(f"Quantized dtype: {act_fp8.dtype}")  # torch.float8_e4m3fn
print(f"Scales shape: {scales.shape}")  # Per-block scales
</syntaxhighlight>

=== Loading FP8 Models ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load DeepSeek-V3 or other FP8 models
# FP8 kernels are automatically used internally
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-V3-Base",
    max_seq_length=4096,
    # FP8 quantization is handled automatically based on model config
)

# The FP8 kernels handle weight dequantization and matmul fusion
# transparently during forward passes
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:unslothai_unsloth_GPU_CUDA_Environment]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_Dtype_Selection]]
