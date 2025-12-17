# Implementation: huggingface_peft_BOFTLayer

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_peft|https://github.com/huggingface/peft]]
* [[source::Paper|Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization|https://huggingface.co/papers/2311.06243]]
|-
! Domains
| [[domain::Parameter-Efficient Fine-Tuning]], [[domain::Deep Learning]], [[domain::Neural Network Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
Base layer implementation for Butterfly Orthogonal Fine-Tuning (BOFT) that uses butterfly factorization to achieve parameter-efficient fine-tuning with orthogonal transformations.

=== Description ===
BOFTLayer is the base class for implementing BOFT adapters in neural networks. It extends BaseTunerLayer and provides the core functionality for applying butterfly factorization-based orthogonal transformations to pretrained model weights. BOFT decomposes orthogonal matrices using butterfly structure, which significantly reduces the number of trainable parameters while maintaining expressiveness. The layer supports both Linear and Conv2d base layers and includes optional CUDA extensions for fast block diagonal operations.

Key features include:
* Butterfly factorization with configurable block sizes and factors
* Cayley parametrization for maintaining orthogonality
* Multiplicative dropout for regularization
* Optional CUDA acceleration via FastBlockDiag
* Scale parameters for output adjustment
* Support for adapter merging and unmerging

=== Usage ===
Use BOFTLayer when you need parameter-efficient fine-tuning with structured orthogonal transformations. It is particularly effective for adapting large pretrained models while preserving their geometric properties. Best suited for scenarios requiring strong regularization and when computational efficiency is important during both training and inference.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft huggingface_peft]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/boft/layer.py src/peft/tuners/boft/layer.py]
* '''Lines:''' 194-464

=== Signature ===
<syntaxhighlight lang="python">
class BOFTLayer(BaseTunerLayer):
    adapter_layer_names = ("boft_R", "boft_s")
    other_param_names = ("boft_block_size", "boft_block_num", "boft_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        """
        Parameters:
        base_layer: the pretrained model layer (Linear or Conv2d)
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.boft.layer import BOFTLayer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Module || Yes || The pretrained layer to augment (nn.Linear or nn.Conv2d)
|-
| boft_block_size || int || No || Size of each BOFT block (default: 8)
|-
| boft_block_num || int || No || Number of BOFT blocks (default: 0, auto-calculated)
|-
| boft_n_butterfly_factor || int || No || Butterfly factorization depth (default: 0)
|-
| boft_dropout || float || No || Multiplicative dropout probability (default: 0.1)
|-
| init_weights || bool/str || No || Weight initialization strategy (default: True)
|}

=== Outputs ===
{| class="wikitable"
! Name !! Type !! Description
|-
| boft_R || nn.Parameter || Rotation parameters for butterfly blocks (shape: [N, D, H, H])
|-
| boft_s || nn.Parameter || Scaling parameters (shape: [out_features, 1])
|-
| boft_P || torch.Tensor || Permutation matrices for butterfly structure (buffer)
|}

== Core Methods ==

=== update_layer ===
<syntaxhighlight lang="python">
def update_layer(
    self,
    adapter_name,
    boft_block_size,
    boft_block_num,
    boft_n_butterfly_factor,
    boft_dropout,
    init_weights,
    inference_mode: bool = False,
    **kwargs,
):
    """
    Update the layer with trainable BOFT weights.
    Validates parameters, initializes butterfly permutations,
    and creates adapter parameters.
    """
</syntaxhighlight>

=== cayley_batch ===
<syntaxhighlight lang="python">
def cayley_batch(self, data):
    """
    Perform Cayley parametrization on batch of skew-symmetric matrices.

    Args:
        data: Batch of skew-symmetric matrices (b, r, c)

    Returns:
        Orthogonal matrices via Cayley transform
    """
</syntaxhighlight>

=== block_butterfly_perm ===
<syntaxhighlight lang="python">
def block_butterfly_perm(self, n, b, r=3, n_butterfly_factor=1):
    """
    Define permutation matrix for block butterfly permutation.

    Args:
        n: size of the permutation matrix
        b: desired number of blocks
        r: base block size (e.g., 2x2, 3x3)
        n_butterfly_factor: depth of butterfly structure
    """
</syntaxhighlight>

== Usage Examples ==

=== Basic BOFT Layer Creation ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.boft.layer import Linear as BOFTLinear

# Create a base linear layer
base_layer = nn.Linear(768, 768)

# Create BOFT adapter with 8 blocks
boft_layer = BOFTLinear(
    base_layer=base_layer,
    adapter_name="default",
    boft_block_size=96,  # 768 / 96 = 8 blocks
    boft_block_num=0,
    boft_n_butterfly_factor=1,
    boft_dropout=0.1,
    init_weights=True
)

# Forward pass
x = torch.randn(4, 768)
output = boft_layer(x)
print(f"Output shape: {output.shape}")  # [4, 768]
</syntaxhighlight>

=== Using Butterfly Factorization ===
<syntaxhighlight lang="python">
import torch.nn as nn
from peft.tuners.boft.layer import Linear as BOFTLinear

# Create BOFT with butterfly factor for more efficient parameterization
base_layer = nn.Linear(1024, 1024)

boft_layer = BOFTLinear(
    base_layer=base_layer,
    adapter_name="butterfly",
    boft_block_size=64,  # 1024 / 64 = 16 blocks
    boft_block_num=0,
    boft_n_butterfly_factor=2,  # Adds butterfly structure
    boft_dropout=0.0,
    init_weights=True
)

# Check parameter count
total_params = sum(p.numel() for p in boft_layer.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params}")
</syntaxhighlight>

=== Conv2d BOFT Adapter ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.boft.layer import Conv2d as BOFTConv2d

# Create base conv layer
base_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)

# Create BOFT adapter for conv layer
boft_conv = BOFTConv2d(
    base_layer=base_conv,
    adapter_name="conv_adapter",
    boft_block_size=32,
    boft_block_num=0,
    boft_n_butterfly_factor=1,
    boft_dropout=0.1,
    init_weights=True
)

# Forward pass
x = torch.randn(2, 256, 32, 32)
output = boft_conv(x)
print(f"Output shape: {output.shape}")  # [2, 256, 32, 32]
</syntaxhighlight>

=== Merging and Unmerging Adapters ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.boft.layer import Linear as BOFTLinear

base_layer = nn.Linear(512, 512)
boft_layer = BOFTLinear(
    base_layer=base_layer,
    adapter_name="merge_demo",
    boft_block_size=64,
    boft_block_num=0,
    boft_n_butterfly_factor=1,
    boft_dropout=0.0,
    init_weights=True
)

# Train the adapter...
# ...

# Merge adapter weights into base layer for inference
boft_layer.merge(safe_merge=True, adapter_names=["merge_demo"])
print(f"Merged: {boft_layer.merged}")  # True

# Inference with merged weights (no adapter overhead)
x = torch.randn(1, 512)
output = boft_layer(x)

# Unmerge if needed
boft_layer.unmerge()
print(f"Merged after unmerge: {boft_layer.merged}")  # False
</syntaxhighlight>

== Implementation Details ==

=== Butterfly Factorization ===
BOFT uses butterfly factorization to decompose orthogonal matrices into a product of sparse block-diagonal matrices and permutation matrices. This structure reduces parameters from O(d^2) to O(d log d) while maintaining expressiveness.

=== Cayley Parametrization ===
The implementation uses Cayley transform to ensure orthogonality:
* Q = (I - R)(I + R)^-1
* Where R is a skew-symmetric matrix (R^T = -R)
* Guarantees Q^T Q = I (orthogonal property)

=== CUDA Acceleration ===
The layer attempts to load a custom CUDA extension (FastBlockDiag) for efficient block diagonal operations. If unavailable, it falls back to PyTorch's native torch.block_diag, and automatically sets butterfly_factor to 1.

=== Parameter Validation ===
The layer performs extensive validation to ensure:
* Block sizes divide evenly into feature dimensions
* Butterfly factors are compatible with block numbers
* Block numbers are even when using butterfly factorization

== Related Pages ==
* [[uses::Implementation:huggingface_peft_FastBlockDiag]]
* [[uses::Implementation:huggingface_peft_MultiplicativeDropoutLayer]]
* [[related_to::Implementation:huggingface_peft_OFTLayer]]
* [[requires_env::Environment:huggingface_peft_CUDA_Training]]
