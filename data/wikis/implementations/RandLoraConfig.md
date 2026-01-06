= RandLoraConfig =

== Knowledge Sources ==
* Source Repository: https://github.com/huggingface/peft
* Paper: https://huggingface.co/papers/2502.00987
* Related: [https://huggingface.co/papers/2406.02528v1 MatMul-Free Computation]

== Domains ==
* [[NLP]]
* [[PEFT]] (Parameter-Efficient Fine-Tuning)
* [[LoRA]] (Low-Rank Adaptation)
* [[Random Projection]]
* [[Sparse Networks]]

== Overview ==

=== Description ===
RandLoraConfig is the configuration class for storing the configuration of a RandLoraModel. RandLora (Random Low-Rank Adaptation) is a parameter-efficient fine-tuning method that uses fixed random bases (basis_A and basis_B) instead of learned low-rank matrices, with only diagonal scaling matrices (lambda/gamma) being trainable.

Unlike standard LoRA where the rank parameter determines trainable parameters (higher rank = more parameters), RandLora's rank is inversely proportional to trainable parameters: reducing the rank increases trainable parameters. The method supports sparse random bases for potential matmul-free computation.

=== Usage ===
RandLoraConfig is used to configure RandLora layers in a model for parameter-efficient fine-tuning. It provides control over the random basis rank, sparsity, dropout, and scaling parameters.

== Code Reference ==

=== Source Location ===
<code>src/peft/tuners/randlora/config.py</code>

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class RandLoraConfig(PeftConfig):
    def __init__(
        self,
        r: int = 32,
        target_modules: Optional[Union[list[str], str]] = None,
        projection_prng_key: int = 0,
        save_projection: bool = True,
        sparse: bool = False,
        very_sparse: bool = False,
        randlora_dropout: float = 0.0,
        randlora_alpha: int = 640,
        fan_in_fan_out: bool = False,
        bias: str = "none",
        modules_to_save: Optional[list[str]] = None,
        init_weights: bool = True,
        layers_to_transform: Optional[Union[list[int], int]] = None,
        layers_pattern: Optional[str] = None,
    )
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.randlora.config import RandLoraConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| r || int || 32 || Random basis rank dimension (inversely proportional to trainable parameters)
|-
| target_modules || Optional[Union[list[str], str]] || None || Names of modules to apply RandLora to (only linear layers supported)
|-
| projection_prng_key || int || 0 || PRNG init key for basis_A and basis_B initialization
|-
| save_projection || bool || True || Whether to save basis_A/basis_B in state dict (increases checkpoint size but ensures reproducibility)
|-
| sparse || bool || False || Use sparse ternary bases (-1, 0, 1) with attribution probability 1/6 for -1/1, 2/3 for 0
|-
| very_sparse || bool || False || Use highly sparse ternary bases with attribution probability 1/√D for -1/1, (1-2/√D) for 0
|-
| randlora_dropout || float || 0.0 || Dropout probability for RandLora layers
|-
| randlora_alpha || int || 640 || Scaling coefficient (typically 20 times the rank, can cause instability if too high)
|-
| fan_in_fan_out || bool || False || Set to True if layer stores weights as (fan_in, fan_out) like Conv1D in GPT-2
|-
| bias || str || "none" || Bias type: 'none', 'all', or 'randlora_only'
|-
| modules_to_save || Optional[list[str]] || None || Modules apart from RandLora layers to be trainable and saved
|-
| init_weights || bool || True || Whether to initialize RandLora layer weights
|-
| layers_to_transform || Optional[Union[list[int], int]] || None || Specific layer indexes to transform
|-
| layers_pattern || Optional[str] || None || Layer pattern name for layers_to_transform
|}

=== Outputs ===
{| class="wikitable"
! Return Type !! Description
|-
| RandLoraConfig || A configured RandLoraConfig instance with peft_type set to PeftType.RANDLORA
|}

=== Post-Initialization ===
The <code>__post_init__</code> method:
* Sets <code>peft_type</code> to <code>PeftType.RANDLORA</code>
* Converts <code>target_modules</code> to a set if provided as a list
* Warns if <code>save_projection</code> is False about potential reproducibility issues

== Usage Examples ==

=== Basic RandLora Configuration ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("base-model-name")

# Configure RandLora
config = RandLoraConfig(
    r=32,
    target_modules=["q_proj", "v_proj"],
    randlora_alpha=640,
    randlora_dropout=0.1
)

# Apply RandLora
peft_model = get_peft_model(model, config)
print(f"Trainable params: {peft_model.num_parameters(only_trainable=True)}")
</syntaxhighlight>

=== Sparse RandLora ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig, get_peft_model

# Use sparse ternary bases for reduced overfitting
config = RandLoraConfig(
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    sparse=True,  # Enable sparse bases
    randlora_alpha=640,
    save_projection=True
)

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Very Sparse Configuration ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig

# Use very sparse bases for maximum regularization
config = RandLoraConfig(
    r=64,  # Higher rank with very sparse = even more regularization
    target_modules=["q_proj", "k_proj", "v_proj"],
    very_sparse=True,  # Highly sparse bases
    randlora_alpha=1280,  # 20 * 64
    randlora_dropout=0.05
)

# Note: very_sparse may decrease performance but reduces overfitting
</syntaxhighlight>

=== PRNG Key for Reproducibility ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig

# Use consistent PRNG key for reproducible random bases
config = RandLoraConfig(
    r=32,
    target_modules=["q_proj", "v_proj"],
    projection_prng_key=42,  # Fixed seed
    save_projection=False,  # Don't save bases, rely on PRNG key
    randlora_alpha=640
)

# Warning: save_projection=False may cause issues across different systems
</syntaxhighlight>

=== Layer-Specific Application ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig

# Apply RandLora only to specific layers
config = RandLoraConfig(
    r=32,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[0, 1, 2, 3],  # Only first 4 layers
    layers_pattern="model.layers",  # Pattern to identify layers
    randlora_alpha=640
)
</syntaxhighlight>

=== With Bias Training ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig

# Train biases along with RandLora
config = RandLoraConfig(
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="randlora_only",  # Train only RandLora biases
    randlora_alpha=640,
    randlora_dropout=0.1
)

# Alternative: bias="all" trains all biases in the model
</syntaxhighlight>

=== Conv1D Layers (GPT-2) ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# GPT-2 uses Conv1D layers with fan_in_fan_out weight storage
model = AutoModelForCausalLM.from_pretrained("gpt2")

config = RandLoraConfig(
    r=32,
    target_modules=["c_attn", "c_proj"],
    fan_in_fan_out=True,  # Important for GPT-2!
    randlora_alpha=640
)

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Lower Rank for More Parameters ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig

# Counterintuitive: lower rank = MORE trainable parameters in RandLora
config_low_rank = RandLoraConfig(
    r=8,  # Low rank
    target_modules=["q_proj", "v_proj"],
    randlora_alpha=160  # 20 * 8
)

config_high_rank = RandLoraConfig(
    r=64,  # High rank
    target_modules=["q_proj", "v_proj"],
    randlora_alpha=1280  # 20 * 64
)

# config_low_rank will have MORE trainable parameters than config_high_rank
</syntaxhighlight>

=== Full Configuration Example ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig, get_peft_model

config = RandLoraConfig(
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    projection_prng_key=12345,
    save_projection=True,  # Save bases for guaranteed reproducibility
    sparse=False,
    very_sparse=False,
    randlora_dropout=0.1,
    randlora_alpha=640,  # 20 * 32
    fan_in_fan_out=False,
    bias="none",
    modules_to_save=["classifier"],  # Save classifier layer
    init_weights=True,
    layers_to_transform=None,  # Apply to all layers
    layers_pattern=None
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
</syntaxhighlight>

== Key Concepts ==

=== Inverse Rank-Parameter Relationship ===
Unlike standard LoRA, RandLora's rank is inversely proportional to parameters:
* '''Lower r''' = Larger diagonal matrices (lambda/gamma) = More trainable parameters
* '''Higher r''' = Smaller diagonal matrices = Fewer trainable parameters

=== Random Basis Projection ===
* '''basis_A''' and '''basis_B''' are fixed random matrices
* Only diagonal scaling matrices are learned
* PRNG key ensures reproducibility of random bases

=== Sparsity Options ===
* '''Dense''' (default): Standard random matrices
* '''Sparse''': Ternary {-1, 0, 1} with P(-1)=P(1)=1/6, P(0)=2/3
* '''Very Sparse''': Ternary with P(-1)=P(1)=1/√D, P(0)=1-2/√D

=== Scaling Considerations ===
The large default alpha (640) can cause numerical instabilities with high learning rates. Consider:
* Reducing learning rate
* Reducing randlora_alpha
* Using gradient clipping

== Related Pages ==
* [[huggingface_peft_RandLoraModel|RandLoraModel]] - Model implementation
* [[huggingface_peft_RandLoraLayer|RandLoraLayer]] - Layer implementation
* [[huggingface_peft_LoraConfig|LoraConfig]] - Standard LoRA configuration
* [[PEFT]] - Parameter-Efficient Fine-Tuning
* [[LoRA]] - Low-Rank Adaptation

== Categories ==
[[Category:PEFT]]
[[Category:Configuration]]
[[Category:LoRA]]
[[Category:Random Projection]]
[[Category:HuggingFace]]
