# Implementation: huggingface_peft_BOFTConv2d

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_peft|https://github.com/huggingface/peft]]
* [[source::Paper|Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization|https://huggingface.co/papers/2311.06243]]
|-
! Domains
| [[domain::Parameter-Efficient Fine-Tuning]], [[domain::Computer Vision]], [[domain::Convolutional Networks]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
BOFT implementation for 2D convolutional layers with butterfly orthogonal transformations adapted for spatial kernels.

=== Description ===
The Conv2d class implements BOFT specifically for nn.Conv2d layers, extending butterfly factorization to convolutional kernels. It inherits from both nn.Module and BOFTLayer, adapting the BOFT methodology to work with 4D convolutional weights. The implementation flattens the convolutional filter dimensions (in_channels * kernel_height * kernel_width) and applies butterfly orthogonal transformations, then reshapes back to convolutional format.

Key adaptations for Conv2d:
* Computes block size based on flattened filter dimension (in_channels * kernel_size^2)
* Maintains spatial structure during transformation
* Supports all standard Conv2d operations (padding, stride, bias)
* Scale parameters have transposed shape for broadcasting

=== Usage ===
Use BOFTConv2d when fine-tuning convolutional layers in vision models such as ResNets, ViTs with convolutional stems, or any CNN architecture. Particularly effective for image classification, object detection, and segmentation tasks where maintaining spatial inductive biases while adapting is important.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft huggingface_peft]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/boft/layer.py src/peft/tuners/boft/layer.py]
* '''Lines:''' 668-1011

=== Signature ===
<syntaxhighlight lang="python">
class Conv2d(nn.Module, BOFTLayer):
    """BOFT implemented in a Conv2d layer."""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        boft_block_size: int = 8,
        boft_block_num: int = 0,
        boft_n_butterfly_factor: int = 0,
        boft_dropout: float = 0.1,
        init_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        """Initialize BOFT Conv2d layer"""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.boft.layer import Conv2d
# Or through PEFT config:
from peft import get_peft_model, BOFTConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Conv2d || Yes || The pretrained convolutional layer to adapt
|-
| adapter_name || str || Yes || Name identifier for this adapter
|-
| boft_block_size || int || No || Block size for BOFT (default: 8)
|-
| boft_block_num || int || No || Number of blocks (default: 0, auto-calculated)
|-
| boft_n_butterfly_factor || int || No || Butterfly depth (default: 0)
|-
| boft_dropout || float || No || Dropout probability (default: 0.1)
|-
| init_weights || bool/str || No || Initialization strategy (default: True)
|}

=== Outputs ===
{| class="wikitable"
! Name !! Type !! Description
|-
| forward() output || torch.Tensor || Convolved features [batch, out_channels, H_out, W_out]
|-
| get_delta_weight() || tuple[Tensor, Tensor] || Butterfly rotation matrix and transposed scale
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
    Update conv2d layer with BOFT weights.

    Key difference from Linear:
    - Uses conv_filter_dim = in_features * kernel_size[0]^2
    - Scale parameter shape is (1, out_features) instead of (out_features, 1)
    - Permutation matrices sized for flattened conv filters
    """
</syntaxhighlight>

=== forward ===
<syntaxhighlight lang="python">
def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """
    Forward pass with BOFT transformation on conv weights.

    Process:
    1. Build butterfly rotation matrix (same as Linear)
    2. Reshape conv weights to 2D: [out_channels, in_channels*kH*kW]
    3. Apply rotation: R * W^T
    4. Apply scaling
    5. Reshape back to 4D: [out_channels, in_channels, kH, kW]
    6. Execute conv2d with transformed weights
    """
</syntaxhighlight>

=== merge ===
<syntaxhighlight lang="python">
def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    """
    Merge adapter weights into base conv layer.

    Reshapes conv weights to 2D, applies transformation,
    then reshapes back to 4D format.
    """
</syntaxhighlight>

=== get_delta_weight ===
<syntaxhighlight lang="python">
def get_delta_weight(self, adapter) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute delta weight for conv adapter.

    Returns:
        butterfly_oft_mat: Rotation matrix
        boft_s: Transposed scale (1, out_features)
    """
</syntaxhighlight>

== Usage Examples ==

=== Basic Conv2d Adaptation ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.boft.layer import Conv2d as BOFTConv2d

# Create pretrained conv layer
base_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
# ... load pretrained weights ...

# Add BOFT adapter
boft_conv = BOFTConv2d(
    base_layer=base_conv,
    adapter_name="conv_adapter",
    boft_block_size=36,  # 64 * 3 * 3 = 576, good divisor
    boft_block_num=0,
    boft_n_butterfly_factor=1,
    boft_dropout=0.1,
    init_weights=True
)

# Forward pass
x = torch.randn(8, 64, 32, 32)
output = boft_conv(x)
print(f"Output shape: {output.shape}")  # [8, 128, 32, 32]
</syntaxhighlight>

=== ResNet Block with BOFT ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft import get_peft_model, BOFTConfig
from torchvision.models import resnet50

# Load pretrained ResNet
model = resnet50(pretrained=True)

# Configure BOFT for conv layers
config = BOFTConfig(
    boft_block_size=18,  # Works for various kernel sizes
    boft_n_butterfly_factor=1,
    boft_dropout=0.05,
    target_modules=["conv1", "conv2", "conv3"],  # Target conv layers in blocks
    task_type="IMAGE_CLASSIFICATION"
)

# Create PEFT model
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

# Fine-tune on new task
images = torch.randn(4, 3, 224, 224)
outputs = peft_model(images)
print(f"Predictions: {outputs.shape}")  # [4, 1000]
</syntaxhighlight>

=== Vision Transformer Conv Stem ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.boft.layer import Conv2d as BOFTConv2d

# ViT-style patch embedding conv layer
base_patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)

# Add BOFT adapter to patch embedding
boft_patch = BOFTConv2d(
    base_layer=base_patch_embed,
    adapter_name="patch_adapter",
    boft_block_size=48,  # 3 * 16 * 16 = 768
    boft_block_num=0,
    boft_n_butterfly_factor=2,
    boft_dropout=0.0,  # No dropout for embedding
    init_weights=True
)

# Process image patches
image = torch.randn(1, 3, 224, 224)
patches = boft_patch(image)
print(f"Patch embeddings: {patches.shape}")  # [1, 768, 14, 14]

# Flatten for transformer
patches_flat = patches.flatten(2).transpose(1, 2)
print(f"Sequence: {patches_flat.shape}")  # [1, 196, 768]
</syntaxhighlight>

=== Depthwise Separable Convolution ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.boft.layer import Conv2d as BOFTConv2d

# Depthwise conv (groups = in_channels)
depthwise = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256)

# BOFT adapter for depthwise
boft_depthwise = BOFTConv2d(
    base_layer=depthwise,
    adapter_name="depthwise",
    boft_block_size=3,  # 1 * 3 * 3 = 9 per group
    boft_block_num=0,
    boft_n_butterfly_factor=0,
    boft_dropout=0.1,
    init_weights=True
)

# Pointwise conv
pointwise = nn.Conv2d(256, 512, kernel_size=1)

# BOFT adapter for pointwise
boft_pointwise = BOFTConv2d(
    base_layer=pointwise,
    adapter_name="pointwise",
    boft_block_size=32,  # 256 * 1 * 1 = 256
    boft_block_num=0,
    boft_n_butterfly_factor=1,
    boft_dropout=0.1,
    init_weights=True
)

# Forward through both
x = torch.randn(4, 256, 28, 28)
x = boft_depthwise(x)
x = boft_pointwise(x)
print(f"Output: {x.shape}")  # [4, 512, 28, 28]
</syntaxhighlight>

=== Safe Merging for Deployment ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.boft.layer import Conv2d as BOFTConv2d

base_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1)
boft_conv = BOFTConv2d(
    base_layer=base_conv,
    adapter_name="trained",
    boft_block_size=48,
    boft_block_num=0,
    boft_n_butterfly_factor=1,
    boft_dropout=0.0,
    init_weights=True
)

# Train the adapter...
# ...

# Safe merge checks for NaNs
try:
    boft_conv.merge(safe_merge=True, adapter_names=["trained"])
    print("Merge successful, no NaNs detected")
except ValueError as e:
    print(f"Merge failed: {e}")
    # Keep using adapter mode

# Verify merged weights work
x = torch.randn(1, 128, 32, 32)
output = boft_conv(x)
assert torch.isfinite(output).all(), "Output contains NaN/Inf"
print("Merged model verified")
</syntaxhighlight>

== Implementation Details ==

=== Convolutional Filter Dimension ===
The key dimension for BOFT in Conv2d is:
* conv_filter_dim = in_channels * kernel_height * kernel_width
* This is the "input dimension" from the filter perspective
* Block parameters are computed based on this dimension

=== Weight Reshaping Strategy ===
1. Original weight: [out_channels, in_channels, kH, kW]
2. Reshape to: [out_channels, in_channels*kH*kW]
3. Apply butterfly rotation (operates on second dimension)
4. Apply scaling
5. Reshape back to: [out_channels, in_channels, kH, kW]

=== Scale Parameter Orientation ===
Unlike Linear (where scale is [out_features, 1]), Conv2d uses:
* Scale shape: (1, out_features)
* This matches the reshaped weight for broadcasting
* Transposed in get_delta_weight for consistency

=== Block Size Selection ===
Good block sizes for common kernels:
* 3x3 with 64 channels: 576 = 64*3*3, use block_size=18,36,48,etc.
* 3x3 with 256 channels: 2304 = 256*3*3, use block_size=72,96,144,etc.
* 1x1 (pointwise): just in_channels, any valid divisor works

=== Memory Considerations ===
Conv2d adapters have different memory characteristics:
* Smaller kernels (3x3, 5x5) → smaller rotation matrices
* Larger channel counts → more parameters in scale
* Butterfly structure still reduces parameters vs. full fine-tuning

== Related Pages ==
* [[implements::Implementation:huggingface_peft_BOFTLayer]]
* [[uses::Implementation:huggingface_peft_FastBlockDiag]]
* [[uses::Implementation:huggingface_peft_MultiplicativeDropoutLayer]]
* [[related_to::Implementation:huggingface_peft_BOFTLinear]]
* [[related_to::Implementation:huggingface_peft_OFTConv2d]]
