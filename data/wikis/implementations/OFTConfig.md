= OFTConfig =

== Knowledge Sources ==

* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* [https://arxiv.org/abs/2306.07280 OFT: Controlling Text-to-Image Diffusion by Orthogonal Finetuning]

== Domains ==

[[Category:NLP]]
[[Category:PEFT]]
[[Category:Orthogonal_Fine_Tuning]]
[[Category:Configuration]]

== Overview ==

=== Description ===

The <code>OFTConfig</code> class is the configuration dataclass for Orthogonal Fine-Tuning (OFT) models in PEFT. It stores all parameters needed to configure and initialize an OFT model, including rank, block size, target modules, dropout, and various optimization options.

OFT is a parameter-efficient fine-tuning method that applies orthogonal transformations to model weights, preserving their norm while allowing adaptation to new tasks. This configuration class provides comprehensive control over all aspects of OFT behavior.

Key features:
* Comprehensive parameter configuration for OFT
* Validation of parameter combinations (r and oft_block_size)
* Support for both standard and constrained OFT (COFT) variants
* Cayley-Neumann parameterization for computational efficiency
* Version compatibility checking for backwards compatibility
* Module selection via target_modules and exclude_modules

=== Usage ===

This configuration class is used to initialize OFT models through PEFT's <code>get_peft_model</code> function. It defines which modules to target, the rank/block size of the transformation, and various training-time behaviors.

== Code Reference ==

=== Source Location ===

File: <code>src/peft/tuners/oft/config.py</code>

Repository: HuggingFace PEFT (Parameter-Efficient Fine-Tuning)

=== Class: OFTConfig ===

==== Signature ====

<syntaxhighlight lang="python">
@dataclass
class OFTConfig(PeftConfig):
    r: int = 0
    oft_block_size: int = 32
    module_dropout: float = 0.0
    target_modules: Optional[Union[list[str], str]] = None
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "oft_only"] = "none"
    exclude_modules: Optional[Union[list[str], str]] = None
    init_weights: bool = True
    layers_to_transform: Optional[Union[list[int], int]] = None
    layers_pattern: Optional[Union[list[str], str]] = None
    modules_to_save: Optional[list[str]] = None
    coft: bool = False
    eps: float = 6e-5
    block_share: bool = False
    use_cayley_neumann: bool = True
    num_cayley_neumann_terms: int = 5
</syntaxhighlight>

==== Import ====

<syntaxhighlight lang="python">
from peft import OFTConfig
# or
from peft.tuners.oft.config import OFTConfig
</syntaxhighlight>

== I/O Contract ==

=== Configuration Parameters ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| r || int || 0 || OFT rank - number of OFT blocks per injected layer
|-
| oft_block_size || int || 32 || OFT block size across different layers (r × oft_block_size = layer dimension)
|-
| module_dropout || float || 0.0 || Multiplicative dropout probability for randomly setting OFT blocks to identity
|-
| target_modules || Optional[Union[list[str], str]] || None || Names of modules to apply adapter to (regex or list, 'all-linear' for all linear layers)
|-
| fan_in_fan_out || bool || False || Set to True if layer stores weight as (fan_in, fan_out)
|-
| bias || Literal["none", "all", "oft_only"] || "none" || Bias type - which biases to train
|-
| exclude_modules || Optional[Union[list[str], str]] || None || Names of modules to exclude from adaptation
|-
| init_weights || bool || True || Whether to initialize OFT weights
|-
| layers_to_transform || Optional[Union[list[int], int]] || None || Specific layer indices to transform
|-
| layers_pattern || Optional[Union[list[str], str]] || None || Layer pattern name (e.g., 'layers', 'h') for layers_to_transform
|-
| modules_to_save || Optional[list[str]] || None || Modules to set as trainable and save (e.g., classifier heads)
|-
| coft || bool || False || Whether to use constrained OFT variant
|-
| eps || float || 6e-5 || Control strength for COFT - freedom of rotation (only when coft=True)
|-
| block_share || bool || False || Whether to share OFT parameters between blocks
|-
| use_cayley_neumann || bool || True || Whether to use Cayley-Neumann formulation for efficiency
|-
| num_cayley_neumann_terms || int || 5 || Number of terms in Cayley-Neumann approximation
|}

=== Validation Rules ===

The <code>__post_init__</code> method enforces several validation rules:

1. Either <code>r</code> or <code>oft_block_size</code> must be non-zero
2. Only one of <code>r</code> or <code>oft_block_size</code> can be specified (XOR constraint)
3. If <code>layers_pattern</code> is specified, <code>layers_to_transform</code> must also be specified

== Usage Examples ==

=== Basic Configuration ===

<syntaxhighlight lang="python">
from peft import OFTConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Create basic OFT configuration
oft_config = OFTConfig(
    r=8,  # 8 OFT blocks
    target_modules=["q_proj", "v_proj"],
    module_dropout=0.1,
)

# Apply to model
model = AutoModelForCausalLM.from_pretrained("model-name")
peft_model = get_peft_model(model, oft_config)
</syntaxhighlight>

=== Using Block Size Instead of Rank ===

<syntaxhighlight lang="python">
from peft import OFTConfig

# Specify block size instead of rank
# For a layer with 768 dimensions and block_size=32, r will be 768/32 = 24
oft_config = OFTConfig(
    oft_block_size=32,  # Block size
    r=0,  # Must be 0 when using oft_block_size
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
</syntaxhighlight>

=== Constrained OFT (COFT) ===

<syntaxhighlight lang="python">
from peft import OFTConfig

# Configure constrained OFT for better control
oft_config = OFTConfig(
    r=16,
    target_modules="all-linear",  # Target all linear layers
    coft=True,  # Enable constrained OFT
    eps=1e-4,   # Control freedom of rotation
    module_dropout=0.05,
)
</syntaxhighlight>

=== Advanced Configuration with Layer Selection ===

<syntaxhighlight lang="python">
from peft import OFTConfig

# Only transform specific layers
oft_config = OFTConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    exclude_modules=["lm_head"],  # Exclude output layer
    layers_to_transform=[0, 1, 2, 3],  # Only first 4 layers
    layers_pattern="layers",  # Pattern for nn.ModuleList name
    modules_to_save=["classifier"],  # Save classifier separately
)
</syntaxhighlight>

=== Cayley-Neumann Optimization ===

<syntaxhighlight lang="python">
from peft import OFTConfig

# Configure Cayley-Neumann parameterization
oft_config = OFTConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj"],
    use_cayley_neumann=True,  # Enable for computational efficiency
    num_cayley_neumann_terms=7,  # More terms = better orthogonality, slower
    block_share=False,
)
</syntaxhighlight>

=== Block Sharing Configuration ===

<syntaxhighlight lang="python">
from peft import OFTConfig

# Share parameters between blocks for memory efficiency
oft_config = OFTConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    block_share=True,  # Share OFT parameters between blocks
    module_dropout=0.0,  # Dropout not compatible with block_share=True
)
</syntaxhighlight>

=== Diffusion Model Configuration ===

<syntaxhighlight lang="python">
from peft import OFTConfig

# Configuration for diffusion models (e.g., Stable Diffusion)
config_unet = OFTConfig(
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
    module_dropout=0.0,
    init_weights=True,
)
</syntaxhighlight>

== Implementation Details ==

=== Parameter Initialization ===

After validation in <code>__post_init__</code>, the configuration:
1. Sets <code>peft_type</code> to <code>PeftType.OFT</code>
2. Converts <code>target_modules</code> and <code>exclude_modules</code> to sets if they are lists

=== Version Compatibility ===

The <code>check_kwargs</code> class method performs version compatibility checks:

1. Checks for <code>oft_block_size</code> parameter (added in PEFT 0.14.0)
2. For <code>use_cayley_neumann</code>, checks version >= 0.18.0 (parameterization changed for numerical stability)

=== Mutually Exclusive Parameters ===

The configuration enforces that exactly one of <code>r</code> or <code>oft_block_size</code> is non-zero:

<syntaxhighlight lang="python">
if not (self.r != 0) ^ (self.oft_block_size != 0):
    raise ValueError(
        f"You can only specify either r ({self.r}) or "
        f"oft_block_size ({self.oft_block_size}), but not both "
        f"simultaneously, because r x oft_block_size == in_features."
    )
</syntaxhighlight>

=== Bias Training Options ===

Three options for bias training:
* <code>"none"</code>: No bias training (default)
* <code>"all"</code>: Train all biases in the model
* <code>"oft_only"</code>: Train only biases in OFT-adapted layers

== Related Pages ==

* [[huggingface_peft_OFTModel|OFTModel]] - Main OFT model implementation using this config
* [[huggingface_peft_OFT_AQLM|OFT AQLM Integration]] - AQLM quantized OFT layers
* [[huggingface_peft_OFT_AWQ|OFT AWQ Integration]] - AWQ quantized OFT layers
* [[huggingface_peft_OFT_GPTQ|OFT GPTQ Integration]] - GPTQ quantized OFT layers
* [[huggingface_peft_OFT_EETQ|OFT EETQ Integration]] - EETQ quantized OFT layers
* [[huggingface_peft_OFT_HQQ|OFT HQQ Integration]] - HQQ quantized OFT layers

== See Also ==

* OFT Paper: [https://arxiv.org/abs/2306.07280 Controlling Text-to-Image Diffusion by Orthogonal Finetuning]
* PEFT Documentation: [https://huggingface.co/docs/peft HuggingFace PEFT Docs]
* Configuration Guide: [https://huggingface.co/docs/peft/main/en/package_reference/config PEFT Config Reference]
