= RoadConfig =

== Knowledge Sources ==
* Source Repository: https://github.com/huggingface/peft
* Paper: https://huggingface.co/papers/2409.00119

== Domains ==
* [[NLP]]
* [[PEFT]] (Parameter-Efficient Fine-Tuning)
* [[Rotation-Based Adaptation]]
* [[Linear Transformations]]
* [[Model Compression]]

== Overview ==

=== Description ===
RoadConfig is the configuration class for storing the configuration of a RoadModel. RoAd (Rotation Adaptation) is a parameter-efficient fine-tuning method that adapts models by applying learned rotations to hidden representations. Unlike additive methods like LoRA, RoAd transforms representations through 2D rotations defined by scale and angle parameters.

The method offers three variants (road_1, road_2, road_4) with increasing parameter counts, allowing users to trade off between efficiency and expressiveness. Elements are grouped into 2D vectors for rotation, with group size affecting inference speed on specialized hardware.

=== Usage ===
RoadConfig is used to configure RoAd layers in a model for parameter-efficient fine-tuning. It specifies the rotation variant, group size for element pairing, and target modules to transform.

== Code Reference ==

=== Source Location ===
<code>src/peft/tuners/road/config.py</code>

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class RoadConfig(PeftConfig):
    def __init__(
        self,
        variant: Union[str, RoadVariant] = "road_1",
        group_size: int = 64,
        init_weights: bool = True,
        target_modules: Optional[Union[list[str], str]] = None,
        modules_to_save: Optional[list[str]] = None,
    )
</syntaxhighlight>

=== RoadVariant Type ===
<syntaxhighlight lang="python">
RoadVariant = Literal["road_1", "road_2", "road_4"]
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.road.config import RoadConfig, RoadVariant
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| variant || Union[str, RoadVariant] || "road_1" || Variant of RoAd: "road_1" (same scale/angle for all pairs), "road_2" (per-element scale/angle), or "road_4" (two scale/angle pairs per element)
|-
| group_size || int || 64 || How elements are grouped into 2D vectors for rotation. Element i pairs with element i+group_size/2. Must be divisible by 2. Higher values (32-64) recommended for speed.
|-
| init_weights || bool || True || Whether to initialize RoAd layer weights
|-
| target_modules || Optional[Union[list[str], str]] || None || Names of modules to apply RoAd to. Can be list, regex, or "all-linear". If None, chosen by model architecture.
|-
| modules_to_save || Optional[list[str]] || None || Modules apart from RoAd layers to be trainable and saved
|}

=== Outputs ===
{| class="wikitable"
! Return Type !! Description
|-
| RoadConfig || A configured RoadConfig instance with peft_type set to PeftType.ROAD
|}

=== Validation ===
The <code>__post_init__</code> method validates:
* <code>variant</code> must be one of: "road_1", "road_2", "road_4"
* <code>group_size</code> must be positive and divisible by 2
* Converts <code>target_modules</code> to set if provided as list

== Variant Comparison ==

{| class="wikitable"
! Variant !! Parameters per Layer !! Scale/Angle Sharing !! Relative Cost
|-
| road_1 || hidden_size || Same for all element pairs || 1×
|-
| road_2 || 2 × hidden_size || Per-element || 2×
|-
| road_4 || 4 × hidden_size || Two different per-element || 4×
|}

== Usage Examples ==

=== Basic RoAd Configuration ===
<syntaxhighlight lang="python">
from peft import RoadConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("base-model-name")

# Configure RoAd with minimal parameters (road_1)
config = RoadConfig(
    variant="road_1",
    target_modules=["q_proj", "v_proj"],
    group_size=64
)

# Apply RoAd
peft_model = get_peft_model(model, config)
print(f"Trainable params: {peft_model.num_parameters(only_trainable=True)}")
</syntaxhighlight>

=== road_2 Variant ===
<syntaxhighlight lang="python">
from peft import RoadConfig, get_peft_model

# Use road_2 for 2x more parameters and expressiveness
config = RoadConfig(
    variant="road_2",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    group_size=64,
    init_weights=True
)

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== road_4 Variant ===
<syntaxhighlight lang="python">
from peft import RoadConfig

# Use road_4 for maximum expressiveness
config = RoadConfig(
    variant="road_4",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    group_size=64
)

# road_4 has 4x the parameters of road_1
</syntaxhighlight>

=== All-Linear Target Modules ===
<syntaxhighlight lang="python">
from peft import RoadConfig, get_peft_model

# Apply RoAd to all linear layers
config = RoadConfig(
    variant="road_1",
    target_modules="all-linear",  # Special keyword
    group_size=64
)

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Regex Pattern for Target Modules ===
<syntaxhighlight lang="python">
from peft import RoadConfig

# Use regex to select specific modules
config = RoadConfig(
    variant="road_2",
    target_modules=".*decoder.*(SelfAttention|EncDecAttention).*(q|v)$",
    group_size=64
)
</syntaxhighlight>

=== Small Model with Smaller Group Size ===
<syntaxhighlight lang="python">
from peft import RoadConfig, get_peft_model

# For very small models, reduce group_size
# Note: hidden_size must be divisible by group_size
config = RoadConfig(
    variant="road_1",
    target_modules=["q_proj", "v_proj"],
    group_size=32,  # Smaller for small models
    init_weights=True
)

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== With Modules to Save ===
<syntaxhighlight lang="python">
from peft import RoadConfig

# Save additional modules beyond RoAd layers
config = RoadConfig(
    variant="road_2",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["classifier", "score"],  # Train these too
    group_size=64
)
</syntaxhighlight>

=== Inference-Optimized Configuration ===
<syntaxhighlight lang="python">
from peft import RoadConfig

# Optimize for VLLM and tensor parallelism
config = RoadConfig(
    variant="road_1",  # Minimal parameters
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    group_size=64,  # Higher group size for better inference speed
    init_weights=True
)

# Note: Ensure hidden_size is divisible by group_size
# For tensor parallelism: (hidden_size // n_partitions) % group_size == 0
</syntaxhighlight>

=== Comparing Variants ===
<syntaxhighlight lang="python">
from peft import RoadConfig, get_peft_model
import copy

base_model = AutoModelForCausalLM.from_pretrained("base-model")

# Test all three variants
variants = ["road_1", "road_2", "road_4"]
models = {}

for variant in variants:
    model = copy.deepcopy(base_model)
    config = RoadConfig(
        variant=variant,
        target_modules=["q_proj", "v_proj"],
        group_size=64
    )
    models[variant] = get_peft_model(model, config)

    trainable = models[variant].num_parameters(only_trainable=True)
    print(f"{variant}: {trainable:,} trainable parameters")

# Output example:
# road_1: 524,288 trainable parameters
# road_2: 1,048,576 trainable parameters  (2x road_1)
# road_4: 2,097,152 trainable parameters  (4x road_1)
</syntaxhighlight>

=== Full Configuration Example ===
<syntaxhighlight lang="python">
from peft import RoadConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

config = RoadConfig(
    variant="road_2",  # Balance between efficiency and performance
    group_size=64,  # Recommended for inference speed
    init_weights=True,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    modules_to_save=None
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
</syntaxhighlight>

== Key Concepts ==

=== Element Grouping ===
Elements are paired for 2D rotation:
* Within each group of size <code>group_size</code>
* Element 0 pairs with element group_size/2
* Element 1 pairs with element group_size/2 + 1
* And so on...

This grouping does not affect model performance (elements are unordered) but affects inference speed on specialized hardware.

=== Rotation Parameters ===
* '''road_1''': Single (scale, angle) for entire layer
* '''road_2''': Per-element (scale, angle)
* '''road_4''': Two (scale, angle) pairs per element

=== Hardware Considerations ===
* Higher <code>group_size</code> (64+) improves inference speed
* Important for VLLM and tensor parallelism
* <code>hidden_size % group_size</code> must equal 0
* With tensor parallelism: <code>(hidden_size // n_partitions) % group_size == 0</code>

=== Rotation vs Additive Methods ===
Unlike LoRA (additive), RoAd applies multiplicative rotations:
* LoRA: <code>output = Wx + ΔWx</code>
* RoAd: <code>output = Rotate(Wx)</code>

== Related Pages ==
* [[huggingface_peft_RoadModel|RoadModel]] - Model implementation
* [[huggingface_peft_RoadLayer|RoadLayer]] - Layer implementation
* [[huggingface_peft_RoadQuantized|RoadQuantized]] - Quantized RoAd layers
* [[huggingface_peft_LoraConfig|LoraConfig]] - Related LoRA configuration
* [[PEFT]] - Parameter-Efficient Fine-Tuning

== Categories ==
[[Category:PEFT]]
[[Category:Configuration]]
[[Category:Rotation-Based Methods]]
[[Category:Model Adaptation]]
[[Category:HuggingFace]]
