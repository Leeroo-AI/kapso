= VeraConfig =

== Knowledge Sources ==

* '''Repository''': [https://github.com/huggingface/peft HuggingFace PEFT]
* '''Paper''': [https://huggingface.co/papers/2310.11454 VeRA: Vector-based Random Matrix Adaptation]
* '''Type''': Configuration Class
* '''Module''': peft.tuners.vera.config

== Domains ==

[[Category:Natural_Language_Processing]]
[[Category:Parameter_Efficient_Fine_Tuning]]
[[Category:Vector_Adaptation]]
[[Category:Low_Rank_Adaptation]]
[[Category:Configuration]]

== Overview ==

=== Description ===

VeraConfig is the configuration class for storing parameters of a VeRA (Vector-based Random Matrix Adaptation) model. VeRA is a parameter-efficient fine-tuning technique that uses far fewer parameters than LoRA by employing shared random projection matrices (vera_A and vera_B) across all layers, with only small trainable scaling vectors (lambda_b and lambda_d) per layer.

The key innovation of VeRA is using higher rank values (default 256) than LoRA while maintaining parameter efficiency through shared random projections initialized with a PRNG key. This allows VeRA to achieve competitive performance with significantly fewer trainable parameters.

=== Usage ===

VeraConfig is used to initialize a VeRA model through the PEFT library. It controls all aspects of the VeRA adaptation including rank, target modules, projection initialization, dropout, and layer-specific transformations.

== Code Reference ==

=== Source Location ===

<code>/tmp/praxium_repo_zyf9ywdz/src/peft/tuners/vera/config.py</code>

=== Signature ===

<syntaxhighlight lang="python">
@dataclass
class VeraConfig(PeftConfig):
    r: int = field(default=256, metadata={"help": "Vera attention dimension"})
    target_modules: Optional[Union[list[str], str]] = field(default=None)
    projection_prng_key: int = field(default=0)
    save_projection: bool = field(default=True)
    vera_dropout: float = field(default=0.0)
    d_initial: float = field(default=0.1)
    fan_in_fan_out: bool = field(default=False)
    bias: str = field(default="none")
    modules_to_save: Optional[list[str]] = field(default=None)
    init_weights: bool = field(default=True)
    layers_to_transform: Optional[Union[list[int], int]] = field(default=None)
    layers_pattern: Optional[Union[list[str], str]] = field(default=None)
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
from peft import VeraConfig
</syntaxhighlight>

== I/O Contract ==

=== Configuration Parameters ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| r || int || 256 || VeRA parameter dimension (rank). Higher values than LoRA ranks recommended
|-
| target_modules || Union[List[str], str] || None || Names or regex of modules to apply VeRA to (only linear layers)
|-
| projection_prng_key || int || 0 || PRNG init key for vera_A and vera_B initialization
|-
| save_projection || bool || True || Whether to save vera_A/vera_B projections in state dict
|-
| vera_dropout || float || 0.0 || Dropout probability for VeRA layers
|-
| d_initial || float || 0.1 || Initial value for vera_lambda_d vector (small values <=0.1 recommended)
|-
| fan_in_fan_out || bool || False || True if layer stores weights as (fan_in, fan_out) like Conv1D
|-
| bias || str || "none" || Bias type: 'none', 'all', or 'vera_only'
|-
| modules_to_save || List[str] || None || Modules besides VeRA layers to set as trainable
|-
| init_weights || bool || True || Whether to initialize weights with default initialization
|-
| layers_to_transform || Union[List[int], int] || None || Specific layer indexes to apply VeRA transformations
|-
| layers_pattern || Union[List[str], str] || None || Layer pattern name for nn.ModuleList (e.g., 'layers', 'h')
|}

=== Validation Rules ===

* If <code>layers_pattern</code> is specified, <code>layers_to_transform</code> must also be specified
* If <code>save_projection</code> is False, a warning is issued about potential restoration issues
* <code>target_modules</code> is converted to a set if provided as a list
* <code>peft_type</code> is automatically set to <code>PeftType.VERA</code>

== Usage Examples ==

=== Basic VeRA Configuration ===

<syntaxhighlight lang="python">
from peft import VeraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Create VeRA config with default settings
config = VeraConfig(
    r=256,
    target_modules=["q_proj", "v_proj"],
    vera_dropout=0.1,
    d_initial=0.1
)

# Apply VeRA to model
peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Advanced Configuration with Layer Selection ===

<syntaxhighlight lang="python">
from peft import VeraConfig, get_peft_model

# Apply VeRA only to specific layers
config = VeraConfig(
    r=512,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    vera_dropout=0.05,
    d_initial=0.05,
    layers_to_transform=[0, 1, 2, 3],  # First 4 layers only
    layers_pattern="layers",
    save_projection=True,
    projection_prng_key=42
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Configuration for GPT-2 Style Models ===

<syntaxhighlight lang="python">
from peft import VeraConfig

# GPT-2 uses Conv1D which stores weights as (fan_in, fan_out)
config = VeraConfig(
    r=256,
    target_modules=["c_attn", "c_proj"],
    fan_in_fan_out=True,  # Important for Conv1D layers
    vera_dropout=0.1,
    bias="vera_only",
    modules_to_save=["classifier"]
)
</syntaxhighlight>

=== Memory-Efficient Configuration ===

<syntaxhighlight lang="python">
from peft import VeraConfig

# Minimize checkpoint size by not saving projections
config = VeraConfig(
    r=256,
    target_modules=["q_proj", "v_proj"],
    save_projection=False,  # Reduces checkpoint size
    projection_prng_key=12345,  # Must be consistent for loading
    vera_dropout=0.0,
    d_initial=0.1
)
</syntaxhighlight>

== Related Pages ==

* [[huggingface_peft_VeraModel|VeraModel]] - Model class that uses this configuration
* [[huggingface_peft_VeraLayer|VeraLayer]] - Layer implementation for VeRA
* [[huggingface_peft_LoraConfig|LoraConfig]] - Similar configuration for LoRA
* [[huggingface_peft_PeftConfig|PeftConfig]] - Base configuration class
* [[Parameter_Efficient_Fine_Tuning|PEFT Overview]]
* [[Low_Rank_Adaptation|LoRA and Variants]]
