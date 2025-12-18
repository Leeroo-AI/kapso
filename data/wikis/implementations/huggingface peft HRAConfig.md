{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|Householder Reflection Adaptation|https://huggingface.co/papers/2405.17484]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Parameter-Efficient Fine-Tuning]], [[domain::Orthogonal Transformation]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
HRAConfig is the configuration class for HRA (Householder Reflection Adaptation), a parameter-efficient fine-tuning method that uses Householder reflections to create orthogonal transformations for model adaptation.

=== Description ===
HRAConfig stores configuration parameters for Householder Reflection Adaptation models. HRA uses Householder reflections to construct orthogonal transformations that adapt pretrained models efficiently. The method maintains orthogonality through optional Gram-Schmidt orthogonalization (apply_GS), which can improve stability and performance.

The configuration is a dataclass that extends PeftConfig and includes validation logic to ensure layer patterns and transformations are properly specified. It supports targeting specific modules, excluding others, and applying transformations to selected layers only.

=== Usage ===
Use HRAConfig when you need to:
* Configure models with orthogonal transformation-based adaptation
* Apply Householder reflections for parameter-efficient fine-tuning
* Control orthogonalization behavior with Gram-Schmidt
* Target specific layers or modules for adaptation
* Fine-tune vision and language models with orthogonal constraints

== Code Reference ==
=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/hra/config.py src/peft/tuners/hra/config.py]
* '''Lines:''' 24-134

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class HRAConfig(PeftConfig):
    r: int = 8
    apply_GS: bool = False
    target_modules: Optional[Union[list[str], str]] = None
    exclude_modules: Optional[Union[list[str], str]] = None
    init_weights: bool = True
    layers_to_transform: Optional[Union[list[int], int]] = None
    layers_pattern: Optional[Union[list[str], str]] = None
    bias: str = "none"
    modules_to_save: Optional[list[str]] = None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import HRAConfig
</syntaxhighlight>

== I/O Contract ==
=== Configuration Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| r || int || 8 || Rank of HRA across different layers (best set to even number)
|-
| apply_GS || bool || False || Whether to apply Gram-Schmidt orthogonalization
|-
| target_modules || Union[list[str], str] || None || Module names or regex to target (e.g., ['q', 'v'] or 'all-linear')
|-
| exclude_modules || Union[list[str], str] || None || Module names or regex to exclude from HRA
|-
| init_weights || bool || True || Whether to initialize HRA weights with default initialization
|-
| layers_to_transform || Union[list[int], int] || None || Specific layer indices to transform
|-
| layers_pattern || Union[list[str], str] || None || Layer pattern name (e.g., 'layers', 'h') for layers_to_transform
|-
| bias || str || "none" || Bias type: 'none', 'all', or 'hra_only'
|-
| modules_to_save || list[str] || None || Additional modules to be trainable and saved
|}

=== Validation Rules ===
{| class="wikitable"
! Rule !! Description
|-
| Rank recommendation || r should be set to an even number for default initialization method to work
|-
| Regex incompatibility || Cannot use layers_to_transform when target_modules is a regex string
|-
| Pattern dependency || layers_pattern can only be used when target_modules is not a regex string
|-
| Pattern requirement || When layers_pattern is specified, layers_to_transform must also be specified
|-
| Set conversion || Converts list target_modules and exclude_modules to sets for efficient lookup
|}

=== Output ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| peft_type || PeftType || Set to PeftType.HRA after initialization
|}

== Usage Examples ==
=== Basic HRA Configuration ===
<syntaxhighlight lang="python">
from peft import HRAConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Basic configuration
config = HRAConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    apply_GS=False
)

base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Configuration with Gram-Schmidt Orthogonalization ===
<syntaxhighlight lang="python">
# Enable Gram-Schmidt for better orthogonality
config = HRAConfig(
    r=16,
    apply_GS=True,  # Apply Gram-Schmidt orthogonalization
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    init_weights=True
)
</syntaxhighlight>

=== Excluding Specific Modules ===
<syntaxhighlight lang="python">
# Target all linear layers but exclude specific ones
config = HRAConfig(
    r=8,
    target_modules="all-linear",
    exclude_modules=["lm_head", "classifier"],
    apply_GS=True
)
</syntaxhighlight>

=== Layer-Specific Transformation ===
<syntaxhighlight lang="python">
# Only transform first 4 layers
config = HRAConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[0, 1, 2, 3],
    layers_pattern="layers",  # or "h" for GPT-2
    apply_GS=True
)
</syntaxhighlight>

=== Regex Pattern Targeting ===
<syntaxhighlight lang="python">
# Use regex to target attention layers
config = HRAConfig(
    r=8,
    target_modules=r".*decoder.*(SelfAttention|EncDecAttention).*(q|v)$",
    apply_GS=True
)
</syntaxhighlight>

=== Configuration for Vision Models (Stable Diffusion) ===
<syntaxhighlight lang="python">
# HRA for text encoder
config_te = HRAConfig(
    r=8,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    init_weights=True,
    apply_GS=True
)

# HRA for UNet
config_unet = HRAConfig(
    r=8,
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
    init_weights=True,
    apply_GS=True
)
</syntaxhighlight>

=== Configuration with Bias Training ===
<syntaxhighlight lang="python">
# Train HRA biases only
config = HRAConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    bias="hra_only",  # Only train HRA adapter biases
    apply_GS=True
)

# Train all biases
config_all = HRAConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    bias="all",  # Train all biases
    apply_GS=True
)
</syntaxhighlight>

=== Save Additional Modules ===
<syntaxhighlight lang="python">
# For sequence classification tasks
config = HRAConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    modules_to_save=["classifier"],  # Also train classifier head
    apply_GS=True
)
</syntaxhighlight>

=== Higher Rank Configuration ===
<syntaxhighlight lang="python">
# Higher rank for more capacity (use even numbers)
config = HRAConfig(
    r=16,  # Even number recommended
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    apply_GS=True,
    init_weights=True
)
</syntaxhighlight>

=== Transform Single Layer ===
<syntaxhighlight lang="python">
# Transform only a specific layer
config = HRAConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=5,  # Single integer for one layer
    layers_pattern="layers",
    apply_GS=True
)
</syntaxhighlight>

== Technical Details ==
=== Householder Reflections ===
HRA uses Householder reflections to construct orthogonal transformations:
* Each reflection is defined by a vector of dimension r
* Multiple reflections are composed to create the full transformation
* Orthogonality is preserved through the reflection structure

=== Gram-Schmidt Orthogonalization ===
When apply_GS=True:
* Applies Gram-Schmidt process to maintain orthogonality
* Can improve numerical stability
* May provide better performance in some cases

=== Rank Considerations ===
* Best to use even values for r for default initialization
* Higher r provides more capacity but increases parameters
* Typical values: 8, 16, 32

=== Supported Layers ===
* torch.nn.Linear
* torch.nn.Conv2d

== Related Pages ==
* [[configures::Component:huggingface_peft_HRAModel]]
* [[inherits_from::Configuration:huggingface_peft_PeftConfig]]
* [[uses::Enumeration:huggingface_peft_PeftType]]
* [[related::Configuration:huggingface_peft_LoraConfig]]
* [[related::Configuration:huggingface_peft_AdaLoraConfig]]
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
* [[based_on::Paper:Householder_Reflection_Adaptation]]
