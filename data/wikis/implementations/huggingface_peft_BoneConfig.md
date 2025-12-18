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

Configuration class for Bone (Block Affine) that stores parameters for Householder reflection-based parameter-efficient fine-tuning.

=== Description ===

BoneConfig stores configuration for Bone/BAT fine-tuning. The r parameter sets the rank of the adaptation (best set to even numbers). The init_weights parameter selects between Bone (True) and BAT ("bat") variants. Note: Bone is deprecated and will be removed in PEFT v0.19.0 - use MissConfig instead.

=== Usage ===

Use BoneConfig for Householder reflection-based fine-tuning. For new projects, prefer MissConfig which supersedes Bone. Existing Bone checkpoints can be converted using the provided script.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/bone/config.py src/peft/tuners/bone/config.py]
* '''Lines:''' 1-130

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class BoneConfig(PeftConfig):
    """
    Configuration for Bone (deprecated, use MissConfig).

    Args:
        r: Rank of Bone (best set to even numbers)
        target_modules: Modules to apply Bone to
        init_weights: True for Bone, "bat" for BAT variant
        bias: Bias handling ('none', 'all', 'bone_only')
        modules_to_save: Additional modules to train/save
    """
    r: int = 64
    target_modules: Optional[Union[list[str], str]] = None
    init_weights: bool | Literal["bat"] = True
    bias: str = "none"
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import BoneConfig  # Deprecated, use MissConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| r || int || No || Rank (default: 64, prefer even numbers)
|-
| target_modules || list[str] || No || Modules to adapt
|-
| init_weights || bool/"bat" || No || Variant selection
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| BoneConfig || dataclass || Configuration for get_peft_model
|}

== Usage Examples ==

=== Basic Bone Configuration (Deprecated) ===
<syntaxhighlight lang="python">
from peft import BoneConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Note: Bone is deprecated, use MissConfig instead
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

config = BoneConfig(
    r=64,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== BAT Variant ===
<syntaxhighlight lang="python">
from peft import BoneConfig

# BAT variant uses different initialization
config = BoneConfig(
    r=64,
    init_weights="bat",
    target_modules=["q_proj", "v_proj"],
)
</syntaxhighlight>

=== Migration to MiSS ===
<syntaxhighlight lang="python">
# Convert existing Bone checkpoint to MiSS:
# python scripts/convert-bone-to-miss.py checkpoint_path

# Use MissConfig for new projects
from peft import MissConfig

config = MissConfig(
    r=64,
    target_modules=["q_proj", "v_proj"],
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
