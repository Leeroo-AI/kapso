# Implementation: huggingface_peft_OFTLayer

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_peft|https://github.com/huggingface/peft]]
* [[source::Paper|Controlling Text-to-Image Diffusion by Orthogonal Finetuning|https://arxiv.org/abs/2306.07280]]
|-
! Domains
| [[domain::Parameter-Efficient Fine-Tuning]], [[domain::Deep Learning]], [[domain::Orthogonal Fine-Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
Base layer implementation for Orthogonal Fine-Tuning (OFT) that applies orthogonal transformations to model weights for parameter-efficient adaptation.

=== Description ===
OFTLayer is the base class for implementing OFT adapters in neural networks. It extends BaseTunerLayer and provides functionality for applying orthogonal transformations using block-diagonal orthogonal matrices. Unlike BOFT which uses butterfly factorization, OFT uses direct orthogonal blocks that can optionally be shared across the feature dimension. The layer supports both standard OFT and constrained OFT (COFT) variants.

Key features include:
* Block-diagonal orthogonal transformations
* Cayley parametrization via skew-symmetric matrices
* Support for Cayley-Neumann series approximation
* Optional block sharing for parameter reduction
* Constrained OFT (COFT) with projection operators
* Multiplicative dropout for regularization
* Support for multiple quantization backends

=== Usage ===
Use OFTLayer when you need parameter-efficient fine-tuning with orthogonal constraints but without the butterfly factorization overhead. It is particularly effective for diffusion models, vision tasks, and scenarios where maintaining norm-preserving transformations is critical. COFT variant is useful when you need controlled rotation freedom.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft huggingface_peft]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/layer.py src/peft/tuners/oft/layer.py]
* '''Lines:''' 310-524

=== Signature ===
<syntaxhighlight lang="python">
class OFTLayer(BaseTunerLayer):
    """Implements the OFT layer."""

    adapter_layer_names: tuple[str, ...] = ("oft_R",)
    other_param_names: tuple[str, ...] = ("r", "oft_block_size", "oft_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        """
        Parameters:
        base_layer: the pretrained model layer (Linear, Conv2d, or quantized variants)
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.oft.layer import OFTLayer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Module || Yes || Pretrained layer (Linear, Conv2d, or quantized variants)
|-
| r || int || No || Number of OFT blocks (default: 8)
|-
| oft_block_size || int || No || Size of each block (default: 0, auto-calculated)
|-
| module_dropout || float || No || Multiplicative dropout probability (default: 0.0)
|-
| coft || bool || No || Use constrained OFT variant (default: False)
|-
| eps || float || No || COFT constraint strength (default: 6e-5)
|-
| block_share || bool || No || Share parameters across blocks (default: False)
|-
| use_cayley_neumann || bool || No || Use Cayley-Neumann approximation (default: False)
|-
| num_cayley_neumann_terms || int || No || Number of terms in approximation (default: 5)
|-
| init_weights || bool/str || No || Weight initialization strategy (default: True)
|}

=== Outputs ===
{| class="wikitable"
! Name !! Type !! Description
|-
| oft_R || nn.ModuleDict || OFTRotationModule instances per adapter
|-
| oft_block_size || dict || Block sizes for each adapter
|-
| r || dict || Number of blocks for each adapter
|}

== Core Methods ==

=== update_layer ===
<syntaxhighlight lang="python">
def update_layer(
    self,
    adapter_name,
    r,
    oft_block_size,
    module_dropout,
    coft,
    eps,
    block_share,
    init_weights,
    use_cayley_neumann,
    num_cayley_neumann_terms,
    inference_mode: bool = False,
    **kwargs,
):
    """
    Update layer with trainable OFT weights.

    Validates parameters, creates OFTRotationModule,
    and initializes adapter weights.
    """
</syntaxhighlight>

=== adjust_oft_parameters ===
<syntaxhighlight lang="python">
def adjust_oft_parameters(self, in_features, params):
    """
    Adjust OFT parameters to be divisible by in_features.

    Finds closest divisor to requested parameter value,
    preferring lower values for ties.
    """
</syntaxhighlight>

=== reset_oft_parameters ===
<syntaxhighlight lang="python">
def reset_oft_parameters(self, adapter_name, init_weights):
    """
    Reset OFT parameters.

    If init_weights is True, initializes to zero (identity rotation).
    If False, uses small random initialization.
    """
</syntaxhighlight>

== Usage Examples ==

=== Basic OFT Layer Creation ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.oft.layer import Linear as OFTLinear

# Create base linear layer
base_layer = nn.Linear(768, 768)

# Create OFT adapter
oft_layer = OFTLinear(
    base_layer=base_layer,
    adapter_name="default",
    r=8,  # 8 blocks
    oft_block_size=0,  # Auto: 768/8 = 96
    module_dropout=0.0,
    coft=False,
    block_share=False,
    init_weights=True
)

# Forward pass
x = torch.randn(4, 768)
output = oft_layer(x)
print(f"Output shape: {output.shape}")  # [4, 768]
</syntaxhighlight>

=== Constrained OFT (COFT) ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.oft.layer import Linear as OFTLinear

# Create COFT adapter with controlled rotation freedom
base_layer = nn.Linear(1024, 1024)

coft_layer = OFTLinear(
    base_layer=base_layer,
    adapter_name="coft",
    r=16,
    oft_block_size=0,  # Auto: 1024/16 = 64
    module_dropout=0.0,
    coft=True,  # Enable constrained OFT
    eps=6e-5,  # Control strength
    block_share=False,
    init_weights=True
)

# COFT projects parameters during forward pass
x = torch.randn(8, 1024)
output = coft_layer(x)
print(f"COFT output: {output.shape}")  # [8, 1024]
</syntaxhighlight>

=== Block Sharing for Efficiency ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.oft.layer import Linear as OFTLinear

base_layer = nn.Linear(2048, 2048)

# Without block sharing
oft_no_share = OFTLinear(
    base_layer=base_layer,
    adapter_name="no_share",
    r=32,
    oft_block_size=0,  # 2048/32 = 64
    block_share=False,
    init_weights=True
)

# With block sharing (same rotation for all blocks)
oft_share = OFTLinear(
    base_layer=base_layer,
    adapter_name="share",
    r=32,
    oft_block_size=0,
    block_share=True,  # Share parameters
    init_weights=True
)

# Check parameter counts
no_share_params = oft_no_share.oft_R["no_share"].weight.numel()
share_params = oft_share.oft_R["share"].weight.numel()

print(f"Without sharing: {no_share_params} params")  # 32 * (64*63/2)
print(f"With sharing: {share_params} params")  # 1 * (64*63/2)
print(f"Reduction: {no_share_params / share_params:.1f}x")
</syntaxhighlight>

=== Cayley-Neumann Approximation ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.oft.layer import Linear as OFTLinear

base_layer = nn.Linear(512, 512)

# Standard Cayley (uses matrix solve)
oft_standard = OFTLinear(
    base_layer=base_layer,
    adapter_name="standard",
    r=8,
    oft_block_size=0,
    use_cayley_neumann=False,  # Use exact Cayley
    init_weights=True
)

# Cayley-Neumann approximation (faster)
oft_neumann = OFTLinear(
    base_layer=base_layer,
    adapter_name="neumann",
    r=8,
    oft_block_size=0,
    use_cayley_neumann=True,  # Use approximation
    num_cayley_neumann_terms=5,  # Number of terms
    init_weights=True
)

x = torch.randn(4, 512)
out_standard = oft_standard(x)
out_neumann = oft_neumann(x)

# Outputs should be similar
print(f"Standard output: {out_standard.shape}")
print(f"Neumann output: {out_neumann.shape}")
</syntaxhighlight>

=== Multiple Adapters ===
<syntaxhighlight lang="python">
import torch
from peft import get_peft_model, OFTConfig
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure OFT
config = OFTConfig(
    r=8,
    module_dropout=0.0,
    coft=False,
    target_modules=["c_attn", "c_proj"],
    task_type="CAUSAL_LM"
)

# Create PEFT model
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

# Fine-tune on task...
</syntaxhighlight>

=== Merging for Inference ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.oft.layer import Linear as OFTLinear

base_layer = nn.Linear(768, 768)
oft_layer = OFTLinear(
    base_layer=base_layer,
    adapter_name="trained",
    r=8,
    oft_block_size=0,
    init_weights=True
)

# Train the adapter...
# ...

# Merge for deployment
oft_layer.merge(safe_merge=True, adapter_names=["trained"])
print(f"Merged: {oft_layer.merged}")  # True

# Inference with no adapter overhead
x = torch.randn(1, 768)
output = oft_layer(x)

# Unmerge if needed
oft_layer.unmerge()
print(f"Merged after unmerge: {oft_layer.merged}")  # False
</syntaxhighlight>

== Implementation Details ==

=== Orthogonal Parametrization ===
OFT maintains orthogonality through:
1. Skew-symmetric matrix representation (R^T = -R)
2. Cayley transform: Q = (I - R)(I + R)^-1
3. Guarantees Q^T Q = I

=== Block Structure ===
For input dimension d and r blocks:
* Block size: d / r
* Each block is an orthogonal matrix
* Blocks form block-diagonal structure
* Optional sharing uses same block repeated

=== COFT Projection ===
When coft=True, parameters are projected:
```python
norm_diff = ||R - I||
if norm_diff > eps:
    R = I + eps * (R - I) / norm_diff
```
This constrains rotation freedom.

=== Cayley-Neumann Series ===
Approximation of Cayley transform:
```
Q ≈ I + 2R + 2R^2 + 2R^3 + ... + R^n
```
Faster than exact solve for large matrices.

=== Supported Layer Types ===
* nn.Linear (all sizes)
* nn.Conv2d (with stride/padding)
* Quantized variants: Linear8bitLt, Linear4bit, QuantLinear (various backends)
* Custom layers with in_features/out_features attributes

== Related Pages ==
* [[uses::Implementation:huggingface_peft_OFTRotationModule]]
* [[uses::Implementation:huggingface_peft_MultiplicativeDropoutLayer]]
* [[related_to::Implementation:huggingface_peft_BOFTLayer]]
* [[implemented_by::Implementation:huggingface_peft_OFTLinear]]
* [[implemented_by::Implementation:huggingface_peft_OFTConv2d]]
