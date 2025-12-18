{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|OFT|https://arxiv.org/abs/2306.07280]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Orthogonal_Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Orthogonal Fine-Tuning layer that applies block-diagonal orthogonal transformations to model weights using Cayley parameterization for stable, geometry-preserving adaptation.

=== Description ===

OFTLayer implements orthogonal transformations using block-diagonal matrices. The layer learns skew-symmetric matrices that are transformed via Cayley parameterization into orthogonal matrices. This ensures the transformation is always orthogonal, preserving the geometry of the weight space. The layer supports constrained OFT (COFT) for additional regularization, block sharing for parameter reduction, and multiplicative dropout for regularization during training.

=== Usage ===

Use OFT when you want to preserve the geometric properties of pretrained weights during fine-tuning. OFT is particularly effective for vision models and when you need stable training without the risk of catastrophic forgetting. The block_share option allows using a single orthogonal matrix across all blocks for extreme parameter efficiency.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/layer.py src/peft/tuners/oft/layer.py]
* '''Lines:''' 1-951

=== Signature ===
<syntaxhighlight lang="python">
class OFTLayer(BaseTunerLayer):
    """
    Orthogonal Fine-Tuning layer.

    Attributes:
        oft_R: OFTRotationModule containing orthogonal parameters
        r: Number of blocks per adapter
        oft_block_size: Size of each orthogonal block
        oft_dropout: Multiplicative dropout layers
    """
    adapter_layer_names = ("oft_R",)
    other_param_names = ("r", "oft_block_size", "oft_dropout")

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        oft_block_size: int,
        module_dropout: float,
        coft: bool,
        eps: float,
        block_share: bool,
        init_weights: bool,
        use_cayley_neumann: bool,
        num_cayley_neumann_terms: int,
        inference_mode: bool = False,
        **kwargs,
    ):
        """Create OFT rotation module."""

class OFTRotationModule(nn.Module):
    """Module containing orthogonal rotation parameters and Cayley transform."""
    def __init__(
        self,
        r: int,
        n_elements: int,
        block_size: int,
        in_features: int,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        use_cayley_neumann: bool = True,
        num_cayley_neumann_terms: int = 5,
    ):
        """Initialize rotation module."""

    def get_weight(self) -> torch.Tensor:
        """Compute block-diagonal orthogonal matrix."""

class Linear(nn.Module, OFTLayer):
    """OFT implemented in Linear layer."""

class Conv2d(nn.Module, OFTLayer):
    """OFT implemented in Conv2d layer."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.oft import OFTLayer, OFTConfig, OFTModel
from peft import OFTConfig, get_peft_model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Module || Yes || The pretrained layer (Linear or Conv2d)
|-
| adapter_name || str || Yes || Name for the adapter
|-
| r || int || Yes* || Number of blocks (set one of r or oft_block_size)
|-
| oft_block_size || int || Yes* || Size of each orthogonal block
|-
| coft || bool || No || Use constrained OFT (default: False)
|-
| eps || float || No || COFT constraint strength (default: 6e-5)
|-
| block_share || bool || No || Share rotation across blocks (default: False)
|-
| module_dropout || float || No || Multiplicative dropout probability
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Input rotated by orthogonal transformation
|-
| get_delta_weight() || torch.Tensor || Block-diagonal orthogonal matrix
|}

== Usage Examples ==

=== Basic OFT Configuration ===
<syntaxhighlight lang="python">
from peft import OFTConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure OFT with block size
config = OFTConfig(
    r=8,                   # Number of blocks (alternative to oft_block_size)
    # oft_block_size=512,  # Or specify block size directly
    target_modules=["q_proj", "v_proj"],
    module_dropout=0.0,
    coft=False,
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Constrained OFT (COFT) ===
<syntaxhighlight lang="python">
from peft import OFTConfig, get_peft_model

# Use COFT for additional regularization
config = OFTConfig(
    r=4,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    coft=True,             # Enable constrained OFT
    eps=6e-5,              # Constraint strength (higher = more regularization)
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== OFT with Block Sharing ===
<syntaxhighlight lang="python">
from peft import OFTConfig, get_peft_model

# Share orthogonal matrix across all blocks for minimal parameters
config = OFTConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    block_share=True,      # Use single rotation for all blocks
)

model = get_peft_model(model, config)
# Dramatically fewer parameters with shared blocks
</syntaxhighlight>

=== OFT for Vision Models ===
<syntaxhighlight lang="python">
from peft import OFTConfig, get_peft_model
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)

# OFT works well for vision fine-tuning
config = OFTConfig(
    r=4,
    target_modules=["query", "value"],
    module_dropout=0.1,    # Add dropout regularization
)

model = get_peft_model(model, config)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
