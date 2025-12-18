{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Repo|LyCORIS|https://github.com/KohakuBlueleaf/LyCORIS]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::LyCORIS]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Low-rank Hadamard product adaptation layer that decomposes weight updates using Hadamard (element-wise) products of low-rank matrices for efficient fine-tuning.

=== Description ===

LoHaLayer implements adaptation using the Hadamard product of two low-rank decompositions. Instead of the additive B*A decomposition used in LoRA, LoHa uses (W1a @ W1b) * (W2a @ W2b), where * denotes element-wise multiplication. For convolutional layers, it uses additional tensor decomposition (t1, t2) for more efficient parameterization. This approach can capture different patterns than LoRA while maintaining similar parameter efficiency.

=== Usage ===

Use LoHa when standard LoRA doesn't capture the adaptation patterns well, particularly for diffusion models and image generation tasks. LoHa is part of the LyCORIS family of adapters and is well-suited for Stable Diffusion fine-tuning. It supports rank dropout and module dropout for regularization.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/loha/layer.py src/peft/tuners/loha/layer.py]
* '''Lines:''' 1-445

=== Signature ===
<syntaxhighlight lang="python">
class LoHaLayer(nn.Module, LycorisLayer):
    """
    Low-rank Hadamard product layer.

    Attributes:
        hada_w1_a, hada_w1_b: First pair of low-rank matrices
        hada_w2_a, hada_w2_b: Second pair of low-rank matrices
        hada_t1, hada_t2: Tensor decomposition for Conv layers
    """
    adapter_layer_names = (
        "hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b",
        "hada_t1", "hada_t2"
    )

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        alpha: float,
        rank_dropout: float,
        module_dropout: float,
        init_weights: bool,
        use_effective_conv2d: bool = False,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        """Create LoHa adapter parameters."""

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """Compute Hadamard product of low-rank decompositions."""

class Linear(LoHaLayer):
    """LoHa implemented in Linear layer."""

class Conv2d(LoHaLayer):
    """LoHa implemented in Conv2d layer."""

class Conv1d(LoHaLayer):
    """LoHa implemented in Conv1d layer."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.loha import LoHaLayer, LoHaConfig, LoHaModel
from peft import LoHaConfig, get_peft_model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Module || Yes || The pretrained layer (Linear, Conv2d, or Conv1d)
|-
| adapter_name || str || No || Name for the adapter (default: "default")
|-
| r || int || Yes || Rank for low-rank decomposition
|-
| alpha || float || No || Scaling factor
|-
| rank_dropout || float || No || Dropout probability for rank dimension
|-
| module_dropout || float || No || Probability of disabling adapter during training
|-
| use_effective_conv2d || bool || No || Use tensor decomposition for Conv2d
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Input + Hadamard product adaptation
|-
| get_delta_weight() || torch.Tensor || (W1a @ W1b) * (W2a @ W2b) scaled
|}

== Usage Examples ==

=== Basic LoHa for Diffusion Models ===
<syntaxhighlight lang="python">
from peft import LoHaConfig, get_peft_model
from diffusers import UNet2DConditionModel

# Load Stable Diffusion UNet
unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet"
)

# Configure LoHa (popular for SD fine-tuning)
config = LoHaConfig(
    r=8,
    alpha=16,
    target_modules=["to_q", "to_v", "to_k", "to_out.0"],
    rank_dropout=0.1,
    module_dropout=0.0,
)

# Create PEFT model
unet = get_peft_model(unet, config)
</syntaxhighlight>

=== LoHa with Effective Conv2d ===
<syntaxhighlight lang="python">
from peft import LoHaConfig, get_peft_model

# Use tensor decomposition for convolutions
config = LoHaConfig(
    r=16,
    alpha=32,
    target_modules=["conv1", "conv2"],
    use_effective_conv2d=True,  # More efficient for larger kernels
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== LoHa with Regularization ===
<syntaxhighlight lang="python">
from peft import LoHaConfig, get_peft_model

# Add dropout regularization
config = LoHaConfig(
    r=8,
    alpha=16,
    target_modules=["to_q", "to_v"],
    rank_dropout=0.1,      # Drop rank dimensions
    module_dropout=0.05,   # Occasionally skip entire adapter
)

model = get_peft_model(model, config)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
