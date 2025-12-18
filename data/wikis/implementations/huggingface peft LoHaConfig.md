{{Implementation
|domain=NLP,Computer Vision,PEFT,Parameter-Efficient Fine-Tuning,LoHa,Low-Rank Adaptation
|link=https://github.com/huggingface/peft
}}

== Overview ==

=== Description ===

The <code>LoHaConfig</code> class is a configuration dataclass for Low-Rank Hadamard Product (LoHa) adaptation in PEFT. LoHa is a parameter-efficient fine-tuning method that uses Hadamard products and low-rank decomposition to adapt pretrained models with minimal trainable parameters.

LoHaConfig inherits from <code>LycorisConfig</code> and provides comprehensive configuration options for LoHa adapters, including rank settings, dropout probabilities, target module specification, and layer-specific parameter patterns. The method is partially described in https://huggingface.co/papers/2108.06098 and is particularly effective for both text and image models.

Key configuration parameters include:
* Rank (r) and alpha for scaling
* Rank and module dropout for regularization
* Effective Conv2d decomposition for convolutional layers
* Fine-grained control over target modules and layers
* Support for rank and alpha patterns per layer

=== Usage ===

LoHaConfig is used to configure LoHa adapters before applying them to a model via the PEFT library. It allows users to specify which modules to adapt, how to decompose them, and what hyperparameters to use for training. The configuration is particularly useful for diffusion models, vision transformers, and language models.

== Code Reference ==

=== Source Location ===
* '''Repository:''' huggingface/peft
* '''File Path:''' <code>src/peft/tuners/loha/config.py</code>
* '''Lines:''' 23-144

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class LoHaConfig(LycorisConfig):
    r: int = 8
    alpha: int = 8
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    use_effective_conv2d: bool = False
    target_modules: Optional[Union[list[str], str]] = None
    exclude_modules: Optional[Union[list[str], str]] = None
    init_weights: bool = True
    layers_to_transform: Optional[Union[list[int], int]] = None
    layers_pattern: Optional[Union[list[str], str]] = None
    modules_to_save: Optional[list[str]] = None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.loha.config import LoHaConfig
# or
from peft import LoHaConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| <code>r</code> || <code>int</code> || <code>8</code> || LoHa rank - controls the dimension of the low-rank decomposition
|-
| <code>alpha</code> || <code>int</code> || <code>8</code> || The alpha parameter for LoHa scaling
|-
| <code>rank_dropout</code> || <code>float</code> || <code>0.0</code> || The dropout probability for rank dimension during training
|-
| <code>module_dropout</code> || <code>float</code> || <code>0.0</code> || The dropout probability for disabling LoHa modules during training
|-
| <code>use_effective_conv2d</code> || <code>bool</code> || <code>False</code> || Use parameter effective decomposition for Conv2d (and Conv1d) with ksize > 1 ("Proposition 3" from FedPara paper)
|-
| <code>target_modules</code> || <code>Optional[Union[list[str], str]]</code> || <code>None</code> || List of module names or regex expression of module names to replace with LoHa. Can be 'all-linear' to target all linear layers except output
|-
| <code>exclude_modules</code> || <code>Optional[Union[list[str], str]]</code> || <code>None</code> || List of module names or regex expression to exclude from LoHa adaptation
|-
| <code>init_weights</code> || <code>bool</code> || <code>True</code> || Whether to initialize the weights of the LoHa layers with their default initialization
|-
| <code>layers_to_transform</code> || <code>Optional[Union[list[int], int]]</code> || <code>None</code> || The layer indices to transform. If specified, only these layers will be adapted
|-
| <code>layers_pattern</code> || <code>Optional[Union[list[str], str]]</code> || <code>None</code> || The layer pattern name (e.g., 'layers' or 'h'), used with layers_to_transform
|-
| <code>modules_to_save</code> || <code>Optional[list[str]]</code> || <code>None</code> || List of modules apart from LoHa layers to be set as trainable and saved in the final checkpoint
|}

'''Inherited from LycorisConfig:'''
{| class="wikitable"
! Parameter !! Type !! Description
|-
| <code>rank_pattern</code> || <code>dict</code> || Mapping from layer names/regex to ranks different from default r
|-
| <code>alpha_pattern</code> || <code>dict</code> || Mapping from layer names/regex to alphas different from default alpha
|}

=== Outputs ===

{| class="wikitable"
! Attribute !! Type !! Description
|-
| <code>peft_type</code> || <code>PeftType</code> || Automatically set to <code>PeftType.LOHA</code> in <code>__post_init__</code>
|-
| <code>target_modules</code> || <code>Optional[Union[set[str], str]]</code> || Converted to set if originally a list
|-
| <code>exclude_modules</code> || <code>Optional[Union[set[str], str]]</code> || Converted to set if originally a list
|}

== Usage Examples ==

=== Example 1: Basic LoHa Configuration for Text Models ===
<syntaxhighlight lang="python">
from peft import LoHaConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Create LoHa configuration
config = LoHaConfig(
    r=8,
    alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
)

# Apply to model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = get_peft_model(model, config)
model.print_trainable_parameters()
</syntaxhighlight>

=== Example 2: LoHa for Stable Diffusion (UNet) ===
<syntaxhighlight lang="python">
from diffusers import StableDiffusionPipeline
from peft import LoHaConfig, LoHaModel

# Configuration for UNet with effective conv2d
config_unet = LoHaConfig(
    r=8,
    alpha=32,
    target_modules=[
        "proj_in",
        "proj_out",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
    use_effective_conv2d=True,  # Important for Conv2d layers
)

model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.unet = LoHaModel(model.unet, config_unet, "default")
</syntaxhighlight>

=== Example 3: LoHa with Dropout and Layer-Specific Ranks ===
<syntaxhighlight lang="python">
from peft import LoHaConfig, get_peft_model

config = LoHaConfig(
    r=8,
    alpha=16,
    target_modules=[".*attention.*(q|k|v)_proj"],  # Regex pattern
    rank_dropout=0.1,  # 10% rank dropout
    module_dropout=0.05,  # 5% module dropout
    # Layer-specific rank overrides
    rank_pattern={
        "^model.layers.0.*": 16,  # Higher rank for first layer
        "^model.layers.31.*": 4,  # Lower rank for last layer
    },
    # Layer-specific alpha overrides
    alpha_pattern={
        "^model.layers.0.*": 32,
    },
    init_weights=True,
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Example 4: LoHa with Selective Layer Transformation ===
<syntaxhighlight lang="python">
from peft import LoHaConfig, get_peft_model

# Only adapt specific layers
config = LoHaConfig(
    r=8,
    alpha=8,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[0, 1, 2, 3],  # Only first 4 layers
    layers_pattern="layers",  # Pattern name in model architecture
    modules_to_save=["lm_head"],  # Also train lm_head
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Example 5: LoHa with All-Linear Targeting ===
<syntaxhighlight lang="python">
from peft import LoHaConfig, get_peft_model, TaskType

# Target all linear layers automatically
config = LoHaConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    alpha=32,
    target_modules="all-linear",  # Targets all linear layers except output
    exclude_modules=["lm_head"],  # Explicitly exclude specific modules
    rank_dropout=0.1,
    init_weights=True,
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

== Implementation Details ==

=== Post-Initialization Processing ===
The <code>__post_init__</code> method performs:
* Sets <code>peft_type</code> to <code>PeftType.LOHA</code>
* Converts <code>target_modules</code> list to set for efficient lookup
* Converts <code>exclude_modules</code> list to set
* Validates that <code>layers_pattern</code> is only used with <code>layers_to_transform</code>

=== Effective Conv2d Decomposition ===
When <code>use_effective_conv2d=True</code>, LoHa applies "Proposition 3" from the FedPara paper for efficient decomposition of convolutional layers with kernel size > 1. This is particularly useful for vision models.

=== Rank and Alpha Patterns ===
The pattern system allows fine-grained control:
* Uses regex matching to identify layers
* Overrides default r and alpha values per layer
* Enables architecture-specific optimization

== Related Pages ==

* [[huggingface_peft_LoHaModel|LoHaModel]] - Model class that applies LoHa adapters
* [[huggingface_peft_LoHaLayer|LoHaLayer]] - Layer implementation for LoHa
* [[huggingface_peft_LycorisConfig|LycorisConfig]] - Base configuration class for Lycoris methods
* [[huggingface_peft_LoKrConfig|LoKrConfig]] - Configuration for LoKr (related method)
* [[huggingface_peft_LoRAConfig|LoRAConfig]] - Configuration for LoRA (alternative method)
* [[huggingface_peft_get_peft_model|get_peft_model]] - Function to create PEFT models

[[Category:PEFT]]
[[Category:Configuration]]
[[Category:LoHa]]
[[Category:Low-Rank Adaptation]]
[[Category:Parameter-Efficient Fine-Tuning]]
