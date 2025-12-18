{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|Bone|https://arxiv.org/abs/2409.15371]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Householder_Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Model class that applies Bone (Block Affine) by wrapping Linear layers with Householder reflection-based transformations.

=== Description ===

BoneModel extends BaseTuner to apply Bone/BAT adaptation to transformer models. It creates BoneLinear layers for Linear target modules. Only torch.nn.Linear layers are supported. Note: Bone is deprecated and will be removed in PEFT v0.19.0.

=== Usage ===

Use BoneModel for Householder reflection fine-tuning. Created automatically via get_peft_model with BoneConfig. For new projects, use MissModel instead.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/bone/model.py src/peft/tuners/bone/model.py]
* '''Lines:''' 1-127

=== Signature ===
<syntaxhighlight lang="python">
class BoneModel(BaseTuner):
    """
    Creates Bone model (deprecated, use MissModel).

    Args:
        model: Base transformer model
        config: BoneConfig
        adapter_name: Name for the adapter

    Attributes:
        prefix: "bone_"
        tuner_layer_cls: BoneLayer
    """
    prefix: str = "bone_"
    tuner_layer_cls = BoneLayer

    def _create_and_replace(self, bone_config, adapter_name, target, ...):
        """Create or update Bone layers."""

    @staticmethod
    def _create_new_module(bone_config, adapter_name, target, **kwargs):
        """Create BoneLinear module."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.bone import BoneModel  # Deprecated
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || nn.Module || Yes || Base transformer model
|-
| config || BoneConfig || Yes || Bone configuration
|-
| adapter_name || str || No || Adapter name (default: "default")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| BoneModel || nn.Module || Model with Bone layers
|}

== Usage Examples ==

=== Stable Diffusion with Bone ===
<syntaxhighlight lang="python">
from diffusers import StableDiffusionPipeline
from peft import BoneModel, BoneConfig

config_te = BoneConfig(
    r=8,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    init_weights=True,
)

config_unet = BoneConfig(
    r=8,
    target_modules=[
        "proj_in", "proj_out", "to_k", "to_q", "to_v", "to_out.0",
        "ff.net.0.proj", "ff.net.2",
    ],
    init_weights=True,
)

model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.text_encoder = BoneModel(model.text_encoder, config_te, "default")
model.unet = BoneModel(model.unet, config_unet, "default")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
