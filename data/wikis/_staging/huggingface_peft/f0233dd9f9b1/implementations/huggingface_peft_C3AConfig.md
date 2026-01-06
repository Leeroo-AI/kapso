{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Circulant_Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Configuration class for C3A (Circulant Convolution Adaptation) that stores parameters for FFT-based block circulant fine-tuning.

=== Description ===

C3AConfig stores configuration for circulant convolution-based adaptation. The block_size parameter must divide both input and output dimensions of target layers. Larger block sizes reduce parameter count. The init_weights parameter supports gaussian, kaiming_uniform, and xavier_uniform (default) initialization. FFT operations require float32.

=== Usage ===

Use C3AConfig for FFT-based circulant adaptation. Choose block_size as the GCD of all target layer dimensions for best compatibility. Use block_size_pattern for layer-specific block sizes.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/c3a/config.py src/peft/tuners/c3a/config.py]
* '''Lines:''' 1-138

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class C3AConfig(PeftConfig):
    """
    Configuration for C3A (Circulant Convolution Adaptation).

    Args:
        block_size: Block size (must divide in/out dimensions)
        target_modules: Modules to apply C3A to
        bias: Bias handling ('none', 'all', 'c3a_only')
        block_size_pattern: Per-layer block size overrides
        init_weights: Initialization method
    """
    block_size: int = 256
    target_modules: Optional[Union[list[str], str]] = None
    bias: str = "none"
    block_size_pattern: Optional[dict] = field(default_factory=dict)
    init_weights: Optional[Union[bool, Literal["gaussian", "kaiming_uniform", "xavier_uniform"]]] = "xavier_uniform"
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import C3AConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| block_size || int || No || Block size (default: 256)
|-
| target_modules || list[str] || No || Modules to adapt
|-
| init_weights || str || No || Initialization method (default: xavier_uniform)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| C3AConfig || dataclass || Configuration for get_peft_model
|}

== Usage Examples ==

=== Basic C3A Configuration ===
<syntaxhighlight lang="python">
from peft import C3AConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

config = C3AConfig(
    block_size=256,
    target_modules=["q_proj", "v_proj"],
    init_weights="xavier_uniform",
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Per-Layer Block Size ===
<syntaxhighlight lang="python">
from peft import C3AConfig

config = C3AConfig(
    block_size=256,  # Default
    block_size_pattern={
        "model.layers.0.self_attn.k_proj": 1280,  # Override for specific layer
    },
    target_modules=["q_proj", "k_proj", "v_proj"],
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
