# Implementation: huggingface_peft_FastBlockDiag

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_peft|https://github.com/huggingface/peft]]
* [[source::Paper|Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization|https://huggingface.co/papers/2311.06243]]
|-
! Domains
| [[domain::CUDA Optimization]], [[domain::Linear Algebra]], [[domain::Automatic Differentiation]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
Custom PyTorch autograd Function implementing fast CUDA-accelerated block diagonal matrix construction for BOFT.

=== Description ===
FastBlockDiag is a custom autograd Function that provides optimized block diagonal operations using a CUDA extension. It transforms 4D tensors of shape (N, D, H, H) into block diagonal matrices of shape (N, D*H, D*H), where each of the D blocks of size HxH is placed along the diagonal. The CUDA implementation significantly outperforms PyTorch's native torch.block_diag for large tensors, which is critical for efficient BOFT training.

The implementation includes both forward and backward passes:
* Forward: Constructs block diagonal from individual blocks
* Backward: Decomposes gradients back to individual block gradients

The CUDA extension is loaded dynamically via get_fbd_cuda() and compiled using torch.utils.cpp_extension with ninja build system.

=== Usage ===
Use FastBlockDiag when implementing BOFT and CUDA is available. It automatically falls back to torch.block_diag if CUDA compilation fails. This is an internal optimization component and typically not used directly by users - it's invoked automatically within BOFTLayer during forward passes.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft huggingface_peft]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/boft/layer.py src/peft/tuners/boft/layer.py]
* '''Lines:''' 103-139

=== Signature ===
<syntaxhighlight lang="python">
class FastBlockDiag(Function):
    """
    Custom autograd Function for fast block diagonal operation using CUDA.

    Optimized for 4D tensors where last two dimensions are equal,
    representing block diagonal matrices.
    """

    @staticmethod
    def forward(ctx, input):
        """
        Args:
            input (Tensor): Shape (N, D, H, H)
        Returns:
            Tensor: Shape (N, D*H, D*H)
        """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output (Tensor): Shape (N, D*H, D*H)
        Returns:
            Tensor: Shape (N, D, H, H)
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.boft.layer import FastBlockDiag
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
! Name !! Type !! Required !! Description
|-
| input || torch.Tensor || Yes || 4D tensor of shape (N, D, H, H) containing D blocks of size HxH
|-
| ctx || AutogradContext || Yes || Context for saving tensors for backward (automatic)
|}

=== Outputs ===
{| class="wikitable"
! Name !! Type !! Description
|-
| output || torch.Tensor || Block diagonal matrix of shape (N, D*H, D*H)
|-
| grad_input || torch.Tensor || Gradient w.r.t. input of shape (N, D, H, H) [backward only]
|}

== Core Methods ==

=== forward ===
<syntaxhighlight lang="python">
@staticmethod
def forward(ctx, input):
    """
    Forward pass: construct block diagonal matrix.

    Input shape: (N, D, H, H) where:
        N = batch size
        D = number of blocks
        H = block size (height/width)

    Output shape: (N, D*H, D*H)

    The blocks are placed along the diagonal:
    [Block_0    0      0   ]
    [   0    Block_1   0   ]
    [   0       0   Block_D ]
    """
    output = get_fbd_cuda().forward(input)[0]
    ctx.save_for_backward(input)
    return output
</syntaxhighlight>

=== backward ===
<syntaxhighlight lang="python">
@staticmethod
def backward(ctx, grad_output):
    """
    Backward pass: decompose gradient back to blocks.

    Input: grad_output of shape (N, D*H, D*H)
    Output: grad_input of shape (N, D, H, H)

    Extracts the D diagonal blocks from the large matrix gradient.
    """
    (input,) = ctx.saved_tensors
    grad_input = get_fbd_cuda().backward(grad_output, input)[0]
    return grad_input
</syntaxhighlight>

== Usage Examples ==

=== Direct Usage (Internal) ===
<syntaxhighlight lang="python">
import torch
from peft.tuners.boft.layer import FastBlockDiag, get_fbd_cuda

# Check if CUDA extension is available
if get_fbd_cuda() is not None:
    # Create batch of block matrices
    N, D, H = 4, 8, 16  # 4 samples, 8 blocks, 16x16 each
    blocks = torch.randn(N, D, H, H, device='cuda')

    # Apply FastBlockDiag
    block_diag_matrix = FastBlockDiag.apply(blocks)

    print(f"Input shape: {blocks.shape}")  # [4, 8, 16, 16]
    print(f"Output shape: {block_diag_matrix.shape}")  # [4, 128, 128]

    # Verify block diagonal structure
    # Non-diagonal blocks should be zero
    block_size = H
    for i in range(D):
        for j in range(D):
            start_i = i * block_size
            end_i = (i + 1) * block_size
            start_j = j * block_size
            end_j = (j + 1) * block_size

            block = block_diag_matrix[0, start_i:end_i, start_j:end_j]
            if i == j:
                # Diagonal blocks should match input
                assert torch.allclose(block, blocks[0, i])
            else:
                # Off-diagonal blocks should be zero
                assert torch.allclose(block, torch.zeros_like(block))

    print("Block diagonal structure verified!")
else:
    print("CUDA extension not available, fallback to torch.block_diag")
</syntaxhighlight>

=== Usage in BOFT Forward Pass ===
<syntaxhighlight lang="python">
import torch
from peft.tuners.boft.layer import FastBlockDiag, get_fbd_cuda

# This code is from BOFTLayer.forward() implementation
def apply_boft_transformation(boft_R, fbd_cuda_available):
    """
    Example showing how FastBlockDiag is used in BOFT.

    Args:
        boft_R: Tensor of shape (N, D, H, H) - rotation matrices
        fbd_cuda_available: bool indicating CUDA availability
    """
    N, D, H, _ = boft_R.shape

    # Reshape and apply Cayley transform (simplified)
    boft_R_reshaped = boft_R.view(N * D, H, H)
    # ... cayley transform ...
    orth_rotate = boft_R_reshaped.view(N, D, H, H)

    # Choose implementation based on CUDA availability
    if fbd_cuda_available:
        # Fast CUDA path
        block_diagonal = FastBlockDiag.apply(orth_rotate)
    else:
        # Fallback to PyTorch
        orth_rotate = orth_rotate.squeeze(0)
        block_diagonal = torch.block_diag(*torch.unbind(orth_rotate))
        block_diagonal = block_diagonal.unsqueeze(0)

    return block_diagonal

# Example usage
boft_R = torch.randn(1, 8, 16, 16, device='cuda')
result = apply_boft_transformation(boft_R, get_fbd_cuda() is not None)
print(f"Result shape: {result.shape}")  # [1, 128, 128]
</syntaxhighlight>

=== Performance Comparison ===
<syntaxhighlight lang="python">
import torch
import time
from peft.tuners.boft.layer import FastBlockDiag, get_fbd_cuda

if get_fbd_cuda() is not None:
    # Test parameters
    N, D, H = 2, 32, 32  # Larger test case
    blocks = torch.randn(N, D, H, H, device='cuda')

    # Warm up
    _ = FastBlockDiag.apply(blocks)
    _ = torch.block_diag(*torch.unbind(blocks[0]))

    # Time CUDA version
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        result_cuda = FastBlockDiag.apply(blocks)
    torch.cuda.synchronize()
    cuda_time = time.time() - start

    # Time PyTorch version
    blocks_squeeze = blocks.squeeze(0)
    start = time.time()
    for _ in range(100):
        result_torch = torch.block_diag(*torch.unbind(blocks_squeeze))
    torch.cuda.synchronize()
    torch_time = time.time() - start

    print(f"FastBlockDiag (CUDA): {cuda_time:.4f}s")
    print(f"torch.block_diag: {torch_time:.4f}s")
    print(f"Speedup: {torch_time / cuda_time:.2f}x")

    # Verify correctness
    result_torch = result_torch.unsqueeze(0)
    assert torch.allclose(result_cuda[0], result_torch[0], rtol=1e-4)
    print("Results match!")
</syntaxhighlight>

=== Gradient Verification ===
<syntaxhighlight lang="python">
import torch
from torch.autograd import gradcheck
from peft.tuners.boft.layer import FastBlockDiag, get_fbd_cuda

if get_fbd_cuda() is not None:
    # Create small test case
    N, D, H = 1, 4, 8
    blocks = torch.randn(N, D, H, H, device='cuda', dtype=torch.float64, requires_grad=True)

    # Test gradient computation
    test_passed = gradcheck(
        FastBlockDiag.apply,
        blocks,
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3
    )

    if test_passed:
        print("Gradient check passed!")
    else:
        print("Gradient check failed!")

    # Manual backward pass test
    blocks_test = torch.randn(1, 4, 8, 8, device='cuda', requires_grad=True)
    output = FastBlockDiag.apply(blocks_test)
    loss = output.sum()
    loss.backward()

    print(f"Input gradient shape: {blocks_test.grad.shape}")  # [1, 4, 8, 8]
    assert blocks_test.grad is not None
    assert not torch.isnan(blocks_test.grad).any()
    print("Backward pass successful!")
</syntaxhighlight>

== Implementation Details ==

=== CUDA Extension Loading ===
The CUDA extension is loaded via get_fbd_cuda():
* Loads C++/CUDA source files from fbd/ directory
* Uses JIT compilation with torch.utils.cpp_extension.load
* Requires ninja build system
* Patches environment to use gcc/g++ compilers
* Returns None if compilation fails

=== Block Diagonal Structure ===
For input (N, D, H, H), the output (N, D*H, D*H) has structure:
```
[Block[0,0]     0          0       ...     0      ]
[    0      Block[0,1]     0       ...     0      ]
[    0          0      Block[0,2]  ...     0      ]
[   ...        ...        ...      ...    ...     ]
[    0          0          0       ... Block[0,D-1]]
```

=== Memory Layout ===
* Input: Contiguous 4D tensor
* Output: 2D block diagonal (sparse structure but dense storage)
* Intermediate: Minimal allocations in CUDA kernel
* Backward: Direct extraction of diagonal blocks

=== Performance Characteristics ===
* CUDA version: O(D * H^2) time complexity
* PyTorch version: O(D * H^2) but with Python overhead
* Speedup typically 2-10x depending on D and H
* Most beneficial when D > 8 and H > 16

=== Fallback Behavior ===
When CUDA is not available:
```python
orth_rotate = orth_rotate.squeeze(0)
block_diagonal = torch.block_diag(*torch.unbind(orth_rotate))
block_diagonal = block_diagonal.unsqueeze(0)
```
This provides identical functionality with reduced performance.

== Related Pages ==
* [[used_by::Implementation:huggingface_peft_BOFTLayer]]
* [[used_by::Implementation:huggingface_peft_BOFTLinear]]
* [[used_by::Implementation:huggingface_peft_BOFTConv2d]]
* [[requires::Environment:huggingface_peft_CUDA_Training]]
