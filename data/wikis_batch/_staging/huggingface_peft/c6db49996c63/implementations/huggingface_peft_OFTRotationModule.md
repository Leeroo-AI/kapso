# Implementation: huggingface_peft_OFTRotationModule

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_peft|https://github.com/huggingface/peft]]
* [[source::Paper|Controlling Text-to-Image Diffusion by Orthogonal Finetuning|https://arxiv.org/abs/2306.07280]]
|-
! Domains
| [[domain::Orthogonal Transformations]], [[domain::Linear Algebra]], [[domain::Parameter-Efficient Fine-Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
PyTorch module encapsulating orthogonal rotation computation for OFT, including Cayley parametrization and optional COFT projection.

=== Description ===
OFTRotationModule is a nn.Module that stores and computes orthogonal rotation matrices for OFT adapters. It maintains a parameter tensor representing the upper triangular part of skew-symmetric matrices, which are then converted to full orthogonal matrices via Cayley parametrization. The module supports both exact Cayley transform (via matrix solve) and Cayley-Neumann series approximation for faster computation.

Key capabilities:
* Skew-symmetric matrix construction from upper triangular parameters
* Cayley transform with two computational paths (exact/approximate)
* Optional COFT projection for constrained rotations
* Block diagonal construction for multiple rotation blocks
* Special handling for Conv2d through unfold/fold operations
* Efficient gradient computation through autograd

=== Usage ===
OFTRotationModule is typically created and managed by OFTLayer rather than used directly. It encapsulates the mathematical operations needed to maintain orthogonality during training. The module's forward pass applies the rotation transformation to input features, while get_weight() returns the full orthogonal matrix for merging operations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft huggingface_peft]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/layer.py src/peft/tuners/oft/layer.py]
* '''Lines:''' 72-308

=== Signature ===
<syntaxhighlight lang="python">
class OFTRotationModule(nn.Module):
    def __init__(
        self,
        r,
        n_elements,
        block_size,
        in_features,
        coft=False,
        eps=6e-5,
        block_share=False,
        kernel_size=(0, 0),
        use_cayley_neumann=True,
        num_cayley_neumann_terms=5,
    ):
        """
        Args:
            r: Number of rotation blocks
            n_elements: Number of parameters (block_size * (block_size-1) / 2)
            block_size: Size of each orthogonal block
            in_features: Input feature dimension
            coft: Enable constrained OFT projection
            eps: COFT constraint parameter
            block_share: Share rotation across all blocks
            kernel_size: For Conv2d layers
            use_cayley_neumann: Use series approximation
            num_cayley_neumann_terms: Number of terms in approximation
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.oft.layer import OFTRotationModule
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
! Name !! Type !! Required !! Description
|-
| r || int || Yes || Number of rotation blocks (or 1 if block_share=True)
|-
| n_elements || int || Yes || Number of parameters: block_size * (block_size - 1) / 2
|-
| block_size || int || Yes || Size of each orthogonal block
|-
| in_features || int || Yes || Total input feature dimension
|-
| coft || bool || No || Enable COFT projection (default: False)
|-
| eps || float || No || COFT constraint strength (default: 6e-5)
|-
| block_share || bool || No || Share rotation parameters (default: False)
|-
| kernel_size || tuple || No || Conv2d kernel size (default: (0, 0))
|-
| use_cayley_neumann || bool || No || Use approximation (default: True)
|-
| num_cayley_neumann_terms || int || No || Approximation terms (default: 5)
|}

=== Outputs ===
{| class="wikitable"
! Name !! Type !! Description
|-
| forward() output || torch.Tensor || Rotated features (same shape as input)
|-
| get_weight() || torch.Tensor || Full orthogonal matrix [in_features, in_features]
|-
| weight || nn.Parameter || Upper triangular parameters [r, n_elements]
|}

== Core Methods ==

=== forward ===
<syntaxhighlight lang="python">
def forward(self, x):
    """
    Apply rotation to input features.

    Process:
    1. Apply COFT projection if enabled
    2. Compute orthogonal matrices via Cayley transform
    3. For Conv2d: unfold → rotate → fold
    4. For Linear: reshape → rotate → reshape
    5. Return rotated features

    Args:
        x: Input tensor (2D for Linear, 4D for Conv2d)

    Returns:
        Rotated tensor (same shape as input)
    """
</syntaxhighlight>

=== get_weight ===
<syntaxhighlight lang="python">
def get_weight(self):
    """
    Compute full orthogonal rotation matrix.

    Returns block diagonal matrix constructed from
    individual rotation blocks. Used for merging.

    Returns:
        torch.Tensor: [in_features, in_features] orthogonal matrix
    """
</syntaxhighlight>

=== _cayley_batch ===
<syntaxhighlight lang="python">
def _cayley_batch(
    self, Q: torch.Tensor, block_size: int,
    use_cayley_neumann: bool = True,
    num_neumann_terms: int = 5
) -> torch.Tensor:
    """
    Cayley parametrization on batch of skew-symmetric matrices.

    Two computation paths:
    1. Exact: R = (I + Q)^-1 (I - Q)
    2. Approximate: R ≈ I + 2Q + 2Q^2 + ... + Q^n

    Args:
        Q: Upper triangular parameters [batch, n_elements]
        block_size: Size of square blocks
        use_cayley_neumann: Use approximation
        num_neumann_terms: Number of terms

    Returns:
        Orthogonal matrices [batch, block_size, block_size]
    """
</syntaxhighlight>

=== _project_batch ===
<syntaxhighlight lang="python">
def _project_batch(self, Q, eps=1e-5):
    """
    COFT projection operator.

    Projects parameters to constrained space:
    if ||R - I|| > eps:
        R = I + eps * (R - I) / ||R - I||

    Args:
        Q: Upper triangular parameters
        eps: Constraint strength

    Returns:
        Projected parameters
    """
</syntaxhighlight>

=== _block_diagonal ===
<syntaxhighlight lang="python">
def _block_diagonal(self, oft_R: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Construct block diagonal matrix.

    If block_share: repeat single block
    Otherwise: place r different blocks on diagonal

    Args:
        oft_R: Rotation blocks [r, block_size, block_size]
        rank: Number of blocks to create

    Returns:
        Block diagonal [rank*block_size, rank*block_size]
    """
</syntaxhighlight>

=== _unfold / _fold ===
<syntaxhighlight lang="python">
def _unfold(self, x):
    """
    Unfold Conv2d input for rotation.
    Converts [B, C, H, W] → [B*H_out*W_out, C*kH*kW]
    """

def _fold(self, x_unfolded, orig_shape):
    """
    Fold back to Conv2d format.
    Converts [B*H_out*W_out, C*kH*kW] → [B, C, H, W]
    """
</syntaxhighlight>

== Usage Examples ==

=== Direct Module Creation ===
<syntaxhighlight lang="python">
import torch
from peft.tuners.oft.layer import OFTRotationModule

# Create rotation module for Linear layer
in_features = 768
block_size = 96
r = 8  # 768 / 96 = 8 blocks

n_elements = block_size * (block_size - 1) // 2  # Upper triangular

rotation_module = OFTRotationModule(
    r=r,
    n_elements=n_elements,
    block_size=block_size,
    in_features=in_features,
    coft=False,
    block_share=False,
    use_cayley_neumann=True,
    num_cayley_neumann_terms=5
)

# Initialize weights
torch.nn.init.zeros_(rotation_module.weight)

# Apply rotation
x = torch.randn(4, 768)
rotated = rotation_module(x)
print(f"Input: {x.shape}, Output: {rotated.shape}")

# Get full rotation matrix
R = rotation_module.get_weight()
print(f"Rotation matrix: {R.shape}")  # [768, 768]
print(f"Is orthogonal: {torch.allclose(R @ R.T, torch.eye(768), atol=1e-5)}")
</syntaxhighlight>

=== COFT with Projection ===
<syntaxhighlight lang="python">
import torch
from peft.tuners.oft.layer import OFTRotationModule

# Create COFT module
rotation_module = OFTRotationModule(
    r=4,
    n_elements=32 * 31 // 2,  # For 32x32 blocks
    block_size=32,
    in_features=128,
    coft=True,  # Enable COFT
    eps=1e-4,  # Constraint strength
    block_share=False,
    use_cayley_neumann=True,
    num_cayley_neumann_terms=5
)

# Initialize with random values
torch.nn.init.normal_(rotation_module.weight, std=0.1)

# Forward pass applies projection
x = torch.randn(8, 128)
output = rotation_module(x)

# Verify constraint is satisfied
R = rotation_module.get_weight()
I = torch.eye(128)
diff_norm = torch.norm(R - I)
print(f"||R - I|| = {diff_norm:.6f}")
print(f"Constraint (eps): {rotation_module.eps:.6f}")
</syntaxhighlight>

=== Block Sharing ===
<syntaxhighlight lang="python">
import torch
from peft.tuners.oft.layer import OFTRotationModule

in_features = 1024
block_size = 64

# Without block sharing
no_share = OFTRotationModule(
    r=16,  # 16 different blocks
    n_elements=64 * 63 // 2,
    block_size=block_size,
    in_features=in_features,
    block_share=False
)

# With block sharing
share = OFTRotationModule(
    r=1,  # Single block, repeated
    n_elements=64 * 63 // 2,
    block_size=block_size,
    in_features=in_features,
    block_share=True
)

print(f"No share params: {no_share.weight.numel()}")  # 16 * 2016
print(f"Share params: {share.weight.numel()}")  # 1 * 2016

# Both produce same output shape
x = torch.randn(4, 1024)
out_no_share = no_share(x)
out_share = share(x)
print(f"Outputs: {out_no_share.shape}, {out_share.shape}")
</syntaxhighlight>

=== Conv2d Rotation ===
<syntaxhighlight lang="python">
import torch
from peft.tuners.oft.layer import OFTRotationModule

# For Conv2d with kernel_size=3, in_channels=64
in_features = 64
kernel_size = 3
conv_filter_dim = in_features * kernel_size * kernel_size  # 576
block_size = 72
r = conv_filter_dim // block_size  # 8 blocks

rotation_module = OFTRotationModule(
    r=r,
    n_elements=block_size * (block_size - 1) // 2,
    block_size=block_size,
    in_features=conv_filter_dim,
    kernel_size=(kernel_size, kernel_size),  # Important for Conv2d
    block_share=False,
    use_cayley_neumann=True,
    num_cayley_neumann_terms=5
)

# Input: 4D tensor [B, C, H, W]
x = torch.randn(2, 64, 32, 32)
rotated = rotation_module(x)
print(f"Conv input: {x.shape}, output: {rotated.shape}")
</syntaxhighlight>

=== Cayley-Neumann vs Exact ===
<syntaxhighlight lang="python">
import torch
import time
from peft.tuners.oft.layer import OFTRotationModule

in_features = 512
block_size = 64
r = 8

# Exact Cayley
exact_module = OFTRotationModule(
    r=r,
    n_elements=block_size * (block_size - 1) // 2,
    block_size=block_size,
    in_features=in_features,
    use_cayley_neumann=False  # Exact
)

# Cayley-Neumann approximation
approx_module = OFTRotationModule(
    r=r,
    n_elements=block_size * (block_size - 1) // 2,
    block_size=block_size,
    in_features=in_features,
    use_cayley_neumann=True,  # Approximate
    num_cayley_neumann_terms=5
)

# Compare outputs
x = torch.randn(16, 512)

start = time.time()
out_exact = exact_module(x)
exact_time = time.time() - start

start = time.time()
out_approx = approx_module(x)
approx_time = time.time() - start

print(f"Exact time: {exact_time:.4f}s")
print(f"Approx time: {approx_time:.4f}s")
print(f"Output difference: {(out_exact - out_approx).abs().max():.6f}")
</syntaxhighlight>

=== Verifying Orthogonality ===
<syntaxhighlight lang="python">
import torch
from peft.tuners.oft.layer import OFTRotationModule

# Create module
rotation_module = OFTRotationModule(
    r=4,
    n_elements=64 * 63 // 2,
    block_size=64,
    in_features=256,
    use_cayley_neumann=False  # Use exact for verification
)

# Initialize and get weight
torch.nn.init.normal_(rotation_module.weight, std=0.01)
R = rotation_module.get_weight()

# Verify orthogonality: R^T R = I
RTR = R.T @ R
I = torch.eye(256)
error = (RTR - I).abs().max()

print(f"Orthogonality error: {error:.8f}")
assert error < 1e-5, "Matrix is not orthogonal!"

# Verify determinant is ±1
det = torch.linalg.det(R)
print(f"Determinant: {det:.6f}")
assert abs(abs(det) - 1.0) < 1e-5, "Determinant is not ±1!"

print("Orthogonality verified!")
</syntaxhighlight>

== Implementation Details ==

=== Skew-Symmetric Construction ===
Parameters are stored as upper triangular elements:
```python
matrix[i, j] = param for i < j
matrix[j, i] = -param (skew-symmetric)
matrix[i, i] = 0 (diagonal is zero)
```

=== Cayley Transform ===
Two implementations:
1. Exact: Solves (I + Q)R = (I - Q)
2. Approximate: R = I + 2Q + 2Q² + 2Q³ + ... + Qⁿ

Approximate is faster but less precise for large rotations.

=== COFT Projection ===
Applied during forward pass when coft=True:
```python
Q_skew = construct_skew_symmetric(weight)
norm = ||Q_skew - I||
if norm > eps:
    Q_skew = I + eps * (Q_skew - I) / norm
```

=== Conv2d Handling ===
For convolutional layers:
1. Unfold input to extract patches
2. Reshape to [N*patches, filter_dim]
3. Apply rotation
4. Reshape and fold back to spatial format

=== Block Sharing ===
When block_share=True:
* Only 1 rotation block is learned
* Repeated rank times during block diagonal construction
* Reduces parameters by factor of r

== Related Pages ==
* [[used_by::Implementation:huggingface_peft_OFTLayer]]
* [[used_by::Implementation:huggingface_peft_OFTLinear]]
* [[used_by::Implementation:huggingface_peft_OFTConv2d]]
* [[related_to::Implementation:huggingface_peft_BOFTLayer]]
