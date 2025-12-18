= ShiraConfig =

== Knowledge Sources ==
* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* Source: src/peft/tuners/shira/config.py

== Domains ==
* [[Natural Language Processing (NLP)]]
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Low-Rank Adaptation]]
* [[Model Compression]]
* [[Transfer Learning]]

== Overview ==

=== Description ===
ShiraConfig is a configuration class for the Sparse High Rank Adapter (SHiRA) method. SHiRA is a parameter-efficient fine-tuning technique that adapts pre-trained models using sparse high-rank adapters. Unlike LoRA which uses low-rank decomposition, SHiRA maintains the same parameter count as LoRA but uses a sparse masking approach to achieve high-rank adaptation.

The configuration defines how SHiRA layers should be applied to target modules, including the rank parameter r, mask type, random seed for mask generation, and initialization settings.

=== Usage ===
ShiraConfig is used to initialize and configure SHiRA adapters on pre-trained models. It extends PeftConfig and provides SHiRA-specific parameters for controlling the sparsity pattern, initialization strategy, and target modules to adapt. The configuration is typically passed to get_peft_model() to create a SHiRA-adapted model.

== Code Reference ==

=== Source Location ===
File: src/peft/tuners/shira/config.py
Lines: 28-130

=== Class Signature ===
<syntaxhighlight lang="python">
@dataclass
class ShiraConfig(PeftConfig):
    """
    Configuration class for ShiraModel.

    Args:
        r: Number of SHiRA parameters (default: 32)
        mask_type: Type of mask function (default: "random")
        random_seed: Random seed for mask generation (default: None)
        target_modules: Module names or regex to replace with SHiRA
        fan_in_fan_out: Weight storage format flag (default: False)
        init_weights: Initialize to zeros (default: True)
        modules_to_save: Additional trainable modules
    """
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.shira.config import ShiraConfig
# Or via the main PEFT interface
from peft import ShiraConfig
</syntaxhighlight>

== I/O Contract ==

=== Configuration Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| r || int || 32 || Number of SHiRA parameters computed as r(m+n) for m x n tensor
|-
| mask_type || Literal["random"] || "random" || Type of mask function for sparsity pattern
|-
| random_seed || Optional[int] || None || Random seed for torch generator in random_mask
|-
| target_modules || Optional[Union[list[str], str]] || None || Module names or regex to replace with SHiRA
|-
| fan_in_fan_out || bool || False || True if layer stores weight as (fan_in, fan_out)
|-
| init_weights || bool || True || Initialize SHiRA weights to zeros; False uses randn
|-
| modules_to_save || Optional[list[str]] || None || Additional modules to set as trainable
|}

=== Attributes ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| peft_type || PeftType || Set to PeftType.SHIRA in __post_init__
|-
| mask_fn || Callable || Mask function set based on mask_type
|}

== Usage Examples ==

=== Basic Configuration ===
<syntaxhighlight lang="python">
from peft import ShiraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Create SHiRA configuration
config = ShiraConfig(
    r=32,
    target_modules=["q_proj", "v_proj"],
    mask_type="random",
    random_seed=42
)

# Apply SHiRA to the model
model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Custom Target Modules with Regex ===
<syntaxhighlight lang="python">
config = ShiraConfig(
    r=64,
    target_modules=r".*decoder.*(SelfAttention|EncDecAttention).*(q|v)$",
    init_weights=True,
    modules_to_save=["classifier"]
)
</syntaxhighlight>

=== Configuration with Custom Mask Function ===
<syntaxhighlight lang="python">
import torch

def custom_mask_fn(layer, r, **kwargs):
    """Custom sparse mask function"""
    m, n = layer.weight.shape
    num_params = r * (m + n)
    mask = torch.zeros(m, n, device=layer.weight.device, dtype=layer.weight.dtype)
    # Custom logic to set mask values
    return mask

config = ShiraConfig(r=32, target_modules=["q_proj", "v_proj"])
config.mask_fn = custom_mask_fn
</syntaxhighlight>

=== Fan-in Fan-out Configuration ===
<syntaxhighlight lang="python">
# For models like GPT-2 that use Conv1D
config = ShiraConfig(
    r=32,
    target_modules=["c_attn", "c_proj"],
    fan_in_fan_out=True
)
</syntaxhighlight>

== Related Pages ==
* [[huggingface_peft_ShiraLayer|ShiraLayer]] - Layer implementation for SHiRA adapters
* [[huggingface_peft_ShiraModel|ShiraModel]] - Model class for SHiRA
* [[huggingface_peft_LoraConfig|LoraConfig]] - Configuration for LoRA adapters
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Low-Rank Adaptation]]

[[Category:Machine Learning]]
[[Category:PEFT]]
[[Category:Model Configuration]]
[[Category:HuggingFace]]
