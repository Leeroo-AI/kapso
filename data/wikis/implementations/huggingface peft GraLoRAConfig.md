{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Parameter-Efficient Fine-Tuning]], [[domain::Low-Rank Adaptation]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
GraloraConfig is the configuration class for GraLoRA (Gradient Low-Rank Adaptation), a parameter-efficient fine-tuning method that extends LoRA with block-structured low-rank decomposition for improved expressivity.

=== Description ===
GraloraConfig stores the configuration parameters for GraLoRA models, which partition the low-rank matrices into multiple subblocks determined by the gralora_k parameter. This block structure increases the expressivity of the adapter by a factor of gralora_k while maintaining the same parameter count as standard LoRA with the same rank. The configuration also supports Hybrid GraLoRA, which combines GraLoRA with vanilla LoRA when hybrid_r > 0.

The configuration is a dataclass that extends PeftConfig and sets the peft_type to GRALORA. It includes validation logic to ensure that the rank r is divisible by gralora_k, which is required for proper subblock partitioning.

=== Usage ===
Use GraloraConfig when you need to:
* Configure GraLoRA models with block-structured low-rank adaptation
* Fine-tune models with improved expressivity compared to standard LoRA
* Experiment with hybrid approaches combining GraLoRA and vanilla LoRA
* Control the number of subblocks and their rank distribution
* Set up parameter-efficient fine-tuning with specific target modules

== Code Reference ==
=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/gralora/config.py src/peft/tuners/gralora/config.py]
* '''Lines:''' 22-183

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class GraloraConfig(PeftConfig):
    r: int = 32
    hybrid_r: int = 0
    target_modules: Optional[Union[list[str], str]] = None
    alpha: int = 64
    gralora_dropout: float = 0.0
    gralora_k: int = 2
    fan_in_fan_out: bool = False
    bias: str = "none"
    modules_to_save: Optional[list[str]] = None
    init_weights: bool = True
    layers_to_transform: Optional[Union[list[int], int]] = None
    layers_pattern: Optional[str] = None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import GraloraConfig
</syntaxhighlight>

== I/O Contract ==
=== Configuration Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| r || int || 32 || GraLoRA rank - determines the rank of each subblock (must be divisible by gralora_k)
|-
| hybrid_r || int || 0 || Rank allocated to vanilla LoRA in Hybrid GraLoRA (enables hybrid mode when > 0)
|-
| target_modules || Union[list[str], str] || None || Module names or regex to target (e.g., ['q', 'v'] or 'all-linear')
|-
| alpha || int || 64 || Scaling factor for GraLoRA adapter (scale = alpha / (r + hybrid_r))
|-
| gralora_dropout || float || 0.0 || Dropout probability for the GraLoRA adapter
|-
| gralora_k || int || 2 || Number of subblocks (2 for rank ≤32, 4 for rank ≥64 recommended)
|-
| fan_in_fan_out || bool || False || True if layer stores weights as (fan_in, fan_out) like Conv1D
|-
| bias || str || "none" || Bias type: 'none', 'all', or 'gralora_only'
|-
| modules_to_save || list[str] || None || Additional modules to be trainable and saved
|-
| init_weights || bool || True || Whether to initialize GraLoRA weights with default initialization
|-
| layers_to_transform || Union[list[int], int] || None || Specific layer indices to transform
|-
| layers_pattern || str || None || Layer pattern name for layers_to_transform (e.g., 'layers', 'h')
|}

=== Validation Rules ===
{| class="wikitable"
! Rule !! Description
|-
| r % gralora_k == 0 || Rank must be divisible by gralora_k for valid subblock partitioning
|-
| target_modules conversion || Converts list to set for efficient lookup
|}

=== Output ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| peft_type || PeftType || Set to PeftType.GRALORA after initialization
|}

== Usage Examples ==
=== Basic GraLoRA Configuration ===
<syntaxhighlight lang="python">
from peft import GraloraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Basic configuration with default settings
config = GraloraConfig(
    r=32,  # Must be divisible by gralora_k
    gralora_k=2,  # 2 subblocks
    target_modules=["q_proj", "v_proj"],
    alpha=64
)

base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Higher Rank Configuration ===
<syntaxhighlight lang="python">
# For higher ranks, use gralora_k=4
config = GraloraConfig(
    r=64,  # Higher rank
    gralora_k=4,  # 4 subblocks for better expressivity
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    alpha=128,
    gralora_dropout=0.1
)
</syntaxhighlight>

=== Hybrid GraLoRA Configuration ===
<syntaxhighlight lang="python">
# Combine GraLoRA with vanilla LoRA
config = GraloraConfig(
    r=32,  # GraLoRA rank
    hybrid_r=8,  # Additional vanilla LoRA rank
    gralora_k=2,
    target_modules=["q_proj", "v_proj"],
    alpha=80  # Scale becomes alpha / (r + hybrid_r) = 80/40 = 2.0
)
</syntaxhighlight>

=== Wildcard Target Modules ===
<syntaxhighlight lang="python">
# Target all linear layers
config = GraloraConfig(
    r=32,
    gralora_k=2,
    target_modules="all-linear",  # Matches all linear layers except output
    alpha=64
)
</syntaxhighlight>

=== Regex Pattern Targeting ===
<syntaxhighlight lang="python">
# Use regex to target specific layers
config = GraloraConfig(
    r=32,
    gralora_k=2,
    target_modules=r".*decoder.*(SelfAttention|EncDecAttention).*(q|v)$",
    alpha=64
)
</syntaxhighlight>

=== Layer-Specific Configuration ===
<syntaxhighlight lang="python">
# Only transform specific layers
config = GraloraConfig(
    r=32,
    gralora_k=2,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[0, 1, 2, 3],  # Only first 4 layers
    layers_pattern="layers",
    alpha=64
)
</syntaxhighlight>

=== Configuration with Bias and Dropout ===
<syntaxhighlight lang="python">
# Fine-grained control over bias and dropout
config = GraloraConfig(
    r=32,
    gralora_k=2,
    target_modules=["q_proj", "v_proj"],
    alpha=64,
    bias="gralora_only",  # Train only GraLoRA biases
    gralora_dropout=0.05,
    modules_to_save=["classifier"]  # Also train classifier
)
</syntaxhighlight>

=== Conv1D Layer Configuration ===
<syntaxhighlight lang="python">
# For models using Conv1D (like GPT-2)
config = GraloraConfig(
    r=32,
    gralora_k=2,
    target_modules=["c_attn", "c_proj"],
    fan_in_fan_out=True,  # Required for Conv1D
    alpha=64
)
</syntaxhighlight>

== Technical Details ==
=== GraLoRA Mathematics ===
The GraLoRA adapter partitions the low-rank matrices into k subblocks:
* Total rank: r
* Number of subblocks: gralora_k
* Rank per subblock: r / gralora_k
* Expressivity multiplier: gralora_k
* Scaling factor: alpha / (r + hybrid_r)

=== Hybrid GraLoRA ===
When hybrid_r > 0:
* GraLoRA parameters: r
* Vanilla LoRA parameters: hybrid_r
* Total parameter count: r + hybrid_r
* Combines block-structured and standard low-rank adaptation

== Related Pages ==
* [[configures::Component:huggingface_peft_GraloraModel]]
* [[inherits_from::Configuration:huggingface_peft_PeftConfig]]
* [[related::Configuration:huggingface_peft_LoraConfig]]
* [[related::Configuration:huggingface_peft_AdaLoraConfig]]
* [[uses::Enumeration:huggingface_peft_PeftType]]
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
