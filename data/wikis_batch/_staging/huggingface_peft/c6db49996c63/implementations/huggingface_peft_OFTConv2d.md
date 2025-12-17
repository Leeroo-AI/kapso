# Implementation: huggingface_peft_OFTConv2d

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_peft|https://github.com/huggingface/peft]]
* [[source::Paper|Controlling Text-to-Image Diffusion by Orthogonal Finetuning|https://arxiv.org/abs/2306.07280]]
|-
! Domains
| [[domain::Parameter-Efficient Fine-Tuning]], [[domain::Computer Vision]], [[domain::Convolutional Networks]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
OFT implementation for 2D convolutional layers with orthogonal transformations adapted for spatial convolutions.

=== Description ===
The Conv2d class implements OFT specifically for nn.Conv2d layers, extending orthogonal fine-tuning to convolutional architectures. It inherits from both nn.Module and OFTLayer, applying block-diagonal orthogonal transformations to convolutional features through the OFTRotationModule. The implementation handles the spatial structure of convolutions by using unfold/fold operations within the rotation module, allowing orthogonal transformations to be applied to image patches.

Key features:
* Spatial-aware orthogonal transformations via unfold/fold
* Works with various kernel sizes (3x3, 5x5, etc.)
* Supports standard Conv2d parameters (stride, padding, groups)
* Constraint: dilation must be 1
* Automatic block size calculation based on filter dimensions
* Compatible with all OFT variants (COFT, block sharing, Cayley-Neumann)

=== Usage ===
Use OFTConv2d when fine-tuning convolutional layers in vision models such as ResNets, U-Nets, or diffusion models. Particularly effective for image generation tasks where OFT was originally designed (Stable Diffusion control). Suitable for any CNN-based architecture requiring parameter-efficient adaptation while preserving spatial structure.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft huggingface_peft]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/layer.py src/peft/tuners/oft/layer.py]
* '''Lines:''' 678-924

=== Signature ===
<syntaxhighlight lang="python">
class Conv2d(nn.Module, OFTLayer):
    """OFT implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 8,
        oft_block_size: int = 0,
        fan_in_fan_out: bool = False,
        module_dropout: float = 0.0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        init_weights: Union[bool, str] = True,
        use_cayley_neumann: bool = False,
        num_cayley_neumann_terms: int = 5,
        **kwargs,
    ) -> None:
        """Initialize OFT Conv2d layer"""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.oft.layer import Conv2d
# Or through PEFT config:
from peft import get_peft_model, OFTConfig
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
| r || int || No || Number of OFT blocks (default: 8)
|-
| oft_block_size || int || No || Block size (default: 0, auto-calculated)
|-
| fan_in_fan_out || bool || No || Weight format flag (default: False)
|-
| module_dropout || float || No || Dropout probability (default: 0.0)
|-
| coft || bool || No || Use constrained OFT (default: False)
|-
| eps || float || No || COFT constraint (default: 6e-5)
|-
| block_share || bool || No || Share blocks (default: False)
|-
| init_weights || bool/str || No || Initialization strategy (default: True)
|-
| use_cayley_neumann || bool || No || Use approximation (default: False)
|-
| num_cayley_neumann_terms || int || No || Approximation terms (default: 5)
|}

=== Outputs ===
{| class="wikitable"
! Name !! Type !! Description
|-
| forward() output || torch.Tensor || Convolved features [batch, out_channels, H_out, W_out]
|-
| get_delta_weight() || torch.Tensor || Orthogonal rotation matrix
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
    Update conv2d layer with OFT weights.

    Key differences from Linear:
    - Checks dilation (must be 1)
    - Uses conv_filter_dim = in_channels * kernel_size^2
    - Passes kernel_size to OFTRotationModule
    - Calculates blocks based on filter dimension
    """
</syntaxhighlight>

=== forward ===
<syntaxhighlight lang="python">
def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """
    Forward pass with OFT transformation on conv features.

    Process:
    1. Check if adapters disabled or merged
    2. For each active adapter:
       - Cast input to rotation dtype
       - Apply OFTRotationModule (handles unfold/fold internally)
    3. Pass rotated features to base conv layer
    4. Return in original dtype

    The rotation module handles spatial structure via unfold/fold.
    """
</syntaxhighlight>

=== merge ===
<syntaxhighlight lang="python">
def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    """
    Merge adapter weights into base conv layer.

    Process:
    1. Get orthogonal matrix from rotation module
    2. Reshape conv weights to 2D
    3. Apply rotation
    4. Reshape back to 4D
    5. Update base layer weight
    """
</syntaxhighlight>

=== get_delta_weight ===
<syntaxhighlight lang="python">
def get_delta_weight(self, adapter_name) -> torch.Tensor:
    """
    Get orthogonal rotation matrix for conv adapter.

    Returns:
        torch.Tensor: Rotation matrix for filter dimension
    """
</syntaxhighlight>

== Usage Examples ==

=== Basic Conv2d Adaptation ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.oft.layer import Conv2d as OFTConv2d

# Create pretrained conv layer
base_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
# ... load pretrained weights ...

# Add OFT adapter
oft_conv = OFTConv2d(
    base_layer=base_conv,
    adapter_name="conv_adapter",
    r=0,  # Auto-calculate
    oft_block_size=72,  # 128 * 3 * 3 = 1152, use 72 as divisor
    module_dropout=0.0,
    coft=False,
    block_share=False,
    init_weights=True
)

# Forward pass
x = torch.randn(8, 128, 32, 32)
output = oft_conv(x)
print(f"Output shape: {output.shape}")  # [8, 256, 32, 32]
</syntaxhighlight>

=== Stable Diffusion U-Net Adaptation ===
<syntaxhighlight lang="python">
import torch
from diffusers import StableDiffusionPipeline
from peft import get_peft_model, OFTConfig

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# Configure OFT for U-Net conv layers
oft_config = OFTConfig(
    r=0,
    oft_block_size=16,  # Good for various conv filters
    module_dropout=0.0,
    coft=False,
    target_modules=["conv1", "conv2"],  # Target U-Net conv layers
    task_type="FEATURE_EXTRACTION"
)

# Apply OFT to U-Net
pipe.unet = get_peft_model(pipe.unet, oft_config)
pipe.unet.print_trainable_parameters()

# Fine-tune for controlled generation
# ... training loop ...

# Generate with adapted model
image = pipe("a photo of a cat", num_inference_steps=50).images[0]
image.save("adapted_output.png")
</syntaxhighlight>

=== ResNet Block with OFT ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.oft.layer import Conv2d as OFTConv2d

# ResNet-style bottleneck block
class ResNetBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

# Add OFT to middle conv (3x3)
base_block = ResNetBottleneck(256, 64, 256)

# Adapt the 3x3 conv
oft_conv2 = OFTConv2d(
    base_layer=base_block.conv2,
    adapter_name="resnet_adapt",
    r=0,
    oft_block_size=36,  # 64 * 3 * 3 = 576
    coft=False,
    init_weights=True
)
base_block.conv2 = oft_conv2

# Forward pass
x = torch.randn(4, 256, 56, 56)
out = base_block.conv1(x)
out = base_block.bn1(out)
out = base_block.relu(out)
out = oft_conv2(out)  # OFT-adapted conv
out = base_block.bn2(out)
out = base_block.relu(out)
out = base_block.conv3(out)
out = base_block.bn3(out)
print(f"ResNet block output: {out.shape}")
</syntaxhighlight>

=== Depthwise Separable with OFT ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.oft.layer import Conv2d as OFTConv2d

# Depthwise convolution
depthwise = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256)

oft_depthwise = OFTConv2d(
    base_layer=depthwise,
    adapter_name="depthwise",
    r=0,
    oft_block_size=3,  # 1 * 3 * 3 per group
    module_dropout=0.1,
    init_weights=True
)

# Pointwise convolution
pointwise = nn.Conv2d(256, 512, kernel_size=1)

oft_pointwise = OFTConv2d(
    base_layer=pointwise,
    adapter_name="pointwise",
    r=0,
    oft_block_size=32,  # 256 * 1 * 1
    module_dropout=0.1,
    init_weights=True
)

# Forward through both
x = torch.randn(4, 256, 28, 28)
x = oft_depthwise(x)
x = oft_pointwise(x)
print(f"Output: {x.shape}")  # [4, 512, 28, 28]
</syntaxhighlight>

=== Dilation Check ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.oft.layer import Conv2d as OFTConv2d

# Standard conv - works fine
standard_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1, dilation=1)
oft_standard = OFTConv2d(
    base_layer=standard_conv,
    adapter_name="standard",
    r=8,
    oft_block_size=0
)
print("Standard conv: OK")

# Dilated conv - will raise error
try:
    dilated_conv = nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2)
    oft_dilated = OFTConv2d(
        base_layer=dilated_conv,
        adapter_name="dilated",
        r=8,
        oft_block_size=0
    )
except ValueError as e:
    print(f"Dilated conv error: {e}")
    # "Conv2d with dilation > 1 is not supported by OFT."
</syntaxhighlight>

=== Merging for Deployment ===
<syntaxhighlight lang="python">
import torch
from peft import PeftModel
from torchvision.models import resnet50

# Load pretrained ResNet with OFT adapter
model = resnet50(pretrained=True)
# ... apply OFT and train ...

# Assume we have model with trained adapters
# model = PeftModel(model, oft_config)

# For each Conv2d with OFT adapter
for name, module in model.named_modules():
    if hasattr(module, 'merge') and 'conv' in name:
        print(f"Merging adapter in {name}")
        module.merge(safe_merge=True)

# Now model has integrated weights, no adapter overhead
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"Predictions: {output.shape}")
</syntaxhighlight>

=== Multi-Scale Feature Pyramid ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.oft.layer import Conv2d as OFTConv2d

class FeaturePyramid(nn.Module):
    def __init__(self):
        super().__init__()
        # Different scales with different kernel sizes
        self.conv3x3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.conv1x1 = nn.Conv2d(256, 256, kernel_size=1)

# Add OFT to each
pyramid = FeaturePyramid()

pyramid.conv3x3 = OFTConv2d(
    pyramid.conv3x3, "oft_3x3", r=0, oft_block_size=36  # 256*3*3=2304
)

pyramid.conv5x5 = OFTConv2d(
    pyramid.conv5x5, "oft_5x5", r=0, oft_block_size=100  # 256*5*5=6400
)

pyramid.conv1x1 = OFTConv2d(
    pyramid.conv1x1, "oft_1x1", r=0, oft_block_size=32  # 256*1*1=256
)

# Process multi-scale features
x = torch.randn(2, 256, 64, 64)
feat3 = pyramid.conv3x3(x)
feat5 = pyramid.conv5x5(x)
feat1 = pyramid.conv1x1(x)

print(f"3x3 features: {feat3.shape}")
print(f"5x5 features: {feat5.shape}")
print(f"1x1 features: {feat1.shape}")
</syntaxhighlight>

== Implementation Details ==

=== Convolutional Filter Dimension ===
The effective dimension for OFT in Conv2d:
```python
conv_filter_dim = in_channels * kernel_height * kernel_width
```

This is the "input dimension" from filter perspective.
Block parameters computed based on this dimension.

=== Spatial Transformation via Unfold/Fold ===
OFTRotationModule handles Conv2d specially:
1. Input: [B, C, H, W]
2. Unfold to patches: [B*H_out*W_out, C*kH*kW]
3. Apply rotation
4. Fold back to: [B, C, H_out, W_out]

This allows orthogonal transformation while preserving spatial structure.

=== Dilation Constraint ===
OFT does not support dilated convolutions (dilation > 1) because:
* Unfold/fold operations assume dense spatial connectivity
* Dilation creates gaps that break rotation assumptions
* Check performed in update_layer()

Standard convolutions (dilation=1) work fine.

=== Block Size Selection ===
Good block sizes for common configurations:
* 3x3, 64 channels: 576 = 64*9, use 18, 36, 48, 72, etc.
* 3x3, 256 channels: 2304 = 256*9, use 36, 48, 72, 96, etc.
* 5x5, 64 channels: 1600 = 64*25, use 25, 50, 80, 100, etc.
* 1x1 (pointwise): just in_channels, any divisor works

=== Memory Considerations ===
Conv2d OFT adapters:
* Smaller kernels → smaller rotation matrices
* Larger channel counts → more parameters in scale (if used)
* Spatial unfold increases memory temporarily during forward
* Consider block sharing for very large conv layers

=== Stride and Padding ===
Stride and padding are preserved:
* Rotation applied before convolution
* Base conv layer uses its original stride/padding
* Output spatial dimensions determined by base layer

=== Groups Parameter ===
Grouped convolutions work but affect block size:
* For groups > 1: filter_dim = (in_channels/groups) * kH * kW
* Example: Depthwise (groups=in_channels): filter_dim = kH * kW

=== Quantization Compatibility ===
Like OFTLinear, Conv2d supports:
* Standard FP32/FP16/BF16 layers
* 8-bit quantized convs (if bitsandbytes supports)
* 4-bit quantized convs (if bitsandbytes supports)

Dispatched through OFT config automatically.

== Related Pages ==
* [[implements::Implementation:huggingface_peft_OFTLayer]]
* [[uses::Implementation:huggingface_peft_OFTRotationModule]]
* [[uses::Implementation:huggingface_peft_MultiplicativeDropoutLayer]]
* [[related_to::Implementation:huggingface_peft_BOFTConv2d]]
* [[related_to::Implementation:huggingface_peft_OFTLinear]]
