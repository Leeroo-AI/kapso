{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Block_Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Block-wise orthogonal weight adaptation layer that applies trainable block transformations to model weights with minimal parameters.

=== Description ===

BoneLayer implements block-wise adaptation where the weight matrix is divided into blocks and each block is transformed by a trainable parameter matrix. It supports two modes: standard "bone" mode which adds block-wise corrections, and "bat" (Block Affine Transformation) mode which applies multiplicative block transformations. The method is parameter-efficient as it only learns small block matrices rather than full weight updates.

=== Usage ===

Use BONE when you want a simple block-wise adaptation method that doesn't require complex factorizations. BONE is suitable when the weight matrix can be cleanly divided into blocks and you want direct control over the block size. The "bat" mode is useful when multiplicative transformations are preferred over additive ones.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/bone/layer.py src/peft/tuners/bone/layer.py]
* '''Lines:''' 1-353

=== Signature ===
<syntaxhighlight lang="python">
class BoneLayer(BaseTunerLayer):
    """
    Block orthogonal adaptation layer.

    Attributes:
        bone_block: Trainable block parameters (nn.ParameterDict)
        bone_r: Block rank per adapter (dict)
    """
    adapter_layer_names = ("bone_block",)
    other_param_names = ("bone_r",)

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        init_weights: bool,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        """Create bone adapter with specified rank."""

class BoneLinear(nn.Module, BoneLayer):
    """Bone implemented in a dense layer."""
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        init_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        """
        Initialize BoneLinear layer.

        Args:
            base_layer: The pretrained linear layer
            adapter_name: Name for the adapter
            r: Block rank/size
            init_weights: True for zeros, "bat" for BAT mode, False for random
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.bone import BoneLayer, BoneConfig, BoneModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Module || Yes || The pretrained linear layer to adapt
|-
| adapter_name || str || Yes || Name identifier for the adapter
|-
| r || int || Yes || Block rank/size for the adaptation
|-
| init_weights || bool or str || No || True=zeros, "bat"=BAT mode, False=random
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Input transformed by base layer + block adaptation
|-
| get_delta_weight() || torch.Tensor || Delta weight computed via block operations
|}

== Usage Examples ==

=== Basic BONE Configuration ===
<syntaxhighlight lang="python">
from peft import BoneConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure BONE
config = BoneConfig(
    r=64,                # Block size
    target_modules=["q_proj", "v_proj"],
    init_weights=True,   # Initialize to identity (zeros)
)

# Create PEFT model
model = get_peft_model(model, config)
</syntaxhighlight>

=== BAT Mode ===
<syntaxhighlight lang="python">
from peft import BoneConfig, get_peft_model

# BAT (Block Affine Transformation) mode
config = BoneConfig(
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    init_weights="bat",  # Use multiplicative block transformation
)

model = get_peft_model(model, config)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
