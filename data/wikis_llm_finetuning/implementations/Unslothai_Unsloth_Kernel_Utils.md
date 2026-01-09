# Implementation: Kernel_Utils

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Kernels]], [[domain::Infrastructure]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Core utility functions for Triton kernels including block size calculation, device management, and bitsandbytes integration.

=== Description ===
`utils.py` provides foundational utilities used throughout Unsloth's kernel implementations. It includes Triton configuration helpers, CUDA/XPU stream management, bitsandbytes dequantization bindings, and device-agnostic abstractions for multi-GPU and Intel XPU support.

=== Usage ===
Import utility functions when writing custom Triton kernels or when you need low-level control over quantization operations. Most users won't interact with this directly.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/utils.py unsloth/kernels/utils.py]
* '''Lines:''' 1-1034

=== Key Functions ===
<syntaxhighlight lang="python">
def calculate_settings(n: int) -> Tuple[int, int]:
    """
    Calculate optimal Triton block size and warp count for dimension n.

    Args:
        n: Feature dimension size

    Returns:
        (BLOCK_SIZE, num_warps): Optimal settings for Triton kernel launch

    Raises:
        RuntimeError: If n exceeds MAX_FUSED_SIZE (65536)
    """

def torch_gpu_device(device):
    """
    Context manager for device-specific operations.
    Returns nullcontext for single-GPU, torch.cuda.device for multi-GPU.
    """

# Triton version-specific tanh implementation
triton_tanh = ...  # libdevice.tanh for Triton 3.0+, tl.math.tanh otherwise

# AMP decorators
torch_amp_custom_fwd = ...  # torch.amp.custom_fwd for torch 2.4+
torch_amp_custom_bwd = ...  # torch.amp.custom_bwd for torch 2.4+

# Bitsandbytes operations
cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
cdequantize_blockwise_fp16_nf4 = bnb.functional.lib.cdequantize_blockwise_fp16_nf4
cdequantize_blockwise_bf16_nf4 = bnb.functional.lib.cdequantize_blockwise_bf16_nf4
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.utils import (
    calculate_settings,
    torch_gpu_device,
    triton_tanh,
    torch_amp_custom_fwd,
    torch_amp_custom_bwd,
    MAX_FUSED_SIZE,
)
</syntaxhighlight>

== I/O Contract ==

=== calculate_settings ===
{| class="wikitable"
|-
! Input !! Type !! Description
|-
| n || int || Feature dimension size
|}

{| class="wikitable"
|-
! Output !! Type !! Description
|-
| BLOCK_SIZE || int || Power of 2, capped at 65536
|-
| num_warps || int || 4, 8, 16, or 32 based on BLOCK_SIZE
|}

== Key Constants ==

{| class="wikitable"
|-
! Constant !! Value !! Description
|-
| MAX_FUSED_SIZE || 65536 || Maximum CUDA block size
|-
| CUDA_STREAMS || tuple || Pre-allocated CUDA stream pointers per device
|-
| WEIGHT_BUFFERS || list || Per-device weight buffer storage
|-
| ABSMAX_BUFFERS || list || Per-device absmax buffer storage
|}

== Usage Examples ==

=== Calculate Kernel Settings ===
<syntaxhighlight lang="python">
from unsloth.kernels.utils import calculate_settings

# For a hidden dimension of 4096
BLOCK_SIZE, num_warps = calculate_settings(4096)
print(f"BLOCK_SIZE={BLOCK_SIZE}, num_warps={num_warps}")
# Output: BLOCK_SIZE=4096, num_warps=8
</syntaxhighlight>

=== Multi-GPU Context Manager ===
<syntaxhighlight lang="python">
from unsloth.kernels.utils import torch_gpu_device

device = torch.device("cuda:1")
with torch_gpu_device(device):
    # Operations execute on cuda:1
    my_triton_kernel[grid](...)
</syntaxhighlight>

=== Custom Triton Kernel with Utils ===
<syntaxhighlight lang="python">
from unsloth.kernels.utils import (
    calculate_settings,
    torch_gpu_device,
    triton_tanh,
)
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Use triton_tanh for version compatibility
    out = triton_tanh(x)
    tl.store(out_ptr + offsets, out, mask=mask)

def launch_kernel(x):
    n = x.numel()
    BLOCK_SIZE, num_warps = calculate_settings(min(n, 65536))
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_gpu_device(x.device):
        my_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return out
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
